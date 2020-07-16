from __future__ import print_function

import os
import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets

from PIL import Image

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='DeepLabV3+')
    parser.add_argument('--dataset', type=str, default='COCO',
                        choices=['cifar10', 'cifar100'], help='dataset')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
            opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

# tested
def set_loader(opt):
    # construct data loader
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root='./cocostuff/dataset/test_images',
                                    transform=train_transform,
                                )

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, pin_memory=True, sampler=train_sampler)

    return train_loader

# tested
def set_model(opt):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=False)
    model.eval()
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


## help function to get the pixel values of annotation images
def get_label(index,height,width):
    file = dirs[index]
    this_image = Image.open(path + '/' + file)
    this_image = this_image.resize((height,width))
    data = list(this_image.getdata())
    data = torch.Tensor(data)
    return data

# the loss class 
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        '''
        Args:
            features: hidden vector of shape [size, n_views, feature_dimention].
            labels: ground truth of shape [size].
            mask: contrastive mask of shape [size, size], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        '''
        device = (torch.device('cuda')
                if features.is_cuda
                else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        
        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+1.0e-8)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / torch.add(mask.sum(1),1.0e-8)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        loss.requries_grad = True
        return loss


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    global i
    i = 0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()


    for images,labels in train_loader:
        model(images)
        # GET features: hidden vector of shape [feature_size, n_views, layer_of_features]
        height = feature_vector.shape[1]
        width = feature_vector.shape[2]
        mul = height * width
        features = feature_vector.view(feature_vector.shape[0],mul).transpose(0,1).contiguous().unsqueeze(1)
        features.requires_grad = True

        # GET labels:
        labels = get_label(i,height,width)
        labels = labels.view(-1)
        labels = labels.contiguous().view(-1,1)
        size = labels.shape[0]
        mask = torch.eq(labels, labels.T).float()
        i += 1

        # compute loss [needs features and labels]
        criterion = SupConLoss(temperature=0.05)
        loss = criterion(features, labels,mask)
        loss.requries_grad = True

        # update metric
        losses.update(loss.item(), size)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Train: [{0}][{1}/{2}]\t'
            'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
            epoch, i + 1, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses))
        sys.stdout.flush()


def main():
    # get the list of annotation files
    global path
    global dirs
    path = './cocostuff/dataset/test_annotations'
    dirs = os.listdir(path)

    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)


    # Define a function that will copy the output of a layer
    # attach the function to model
    def copy_data(m, i, o):
        print(m)
        print('input', i)
        print('output', o)
        global feature_vector 
        feature_vector = torch.zeros(i[0][0].shape)
        feature_vector.copy_(i[0][0].data)
        
        global feature_label 
        feature_label = torch.zeros(o[0].shape)
        feature_label.copy_(o[0].data)
    #Attach the function to our selected layer
    layer = model.classifier[4]
    h = layer.register_forward_hook(copy_data)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    #detach the copy function from the model
    h.remove()
    
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()