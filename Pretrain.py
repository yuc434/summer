
from __future__ import print_function

import torch
import torch.nn as nn

from copy import deepcopy
model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

from PIL import Image
from torchvision import transforms
input_image = Image.open("test.JPEG")


'''
function to get features, labels, and masks
    --features: hidden vector of shape [size, layer]
    --labels: ground truth of shape [size]
    --masks: contrastive mask of shape [size,size], mask_{i,j}=1 if sample j 
                has the same class as sample i. Can be asymmetric.
'''
def get_f_l_m(input_image):
    #Define a function that will copy the output of a layer
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

    # normalization transforms
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0) 

    #run the model 
    with torch.no_grad():
        model(input_batch)['out'][0]

    height = feature_vector.shape[1]
    width = feature_vector.shape[2]
    mul = height * width

    #detach the copy function from the model
    h.remove()

    #flatten features and labels 
    features = feature_vector.view(feature_vector.shape[0],mul).transpose(0,1).contiguous()
    labels = feature_label.argmax(0)
    labels = labels.view(-1)
    labels = labels.contiguous().view(-1,1)

    #set the mask 
    mask = torch.eq(labels, labels.T).float()
    return features,labels,mask


# # create a color pallette, selecting a color for each class
# palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
# colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
# colors = (colors % 255).numpy().astype("uint8")

# # plot the semantic segmentation predictions of 21 classes in each color
# r = Image.fromarray(labels.byte().cpu().numpy()).resize(input_image.size)
# r.putpalette(colors)

# import matplotlib.pyplot as plt
# plt.imshow(r)
# plt.show()



'''
loss
'''

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
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
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
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



#test
features,labels,mask = get_f_l_m(input_image)
features = features.unsqueeze(1)
criterion = SupConLoss(temperature=0.05)

loss = criterion(features, labels,mask)






