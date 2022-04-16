'''
Model for YOLOv2 detection + classification.
'''

import torch
import torch.nn as NN
from torch.optim import Adam, lr_scheduler
import cfg

class YOLOv2(NN.Module):
    '''
    Darknet-19 architecture.
    '''

    def _initialize_weights(self):
        '''
        Weight initialization module.
        '''
        for mod in self.modules():
            if isinstance(mod, NN.Conv2d):
                NN.init.kaiming_normal_(mod.weight, mode='fan_out')
                if mod.bias is not None:
                    NN.init.constant_(mod.bias, 0.5)
            elif isinstance(mod, NN.BatchNorm2d):
                NN.init.constant_(mod.weight, 1)
                NN.init.constant_(mod.bias, 0.5)

    def __init__(self, k, num_classes, init_weights=True):

        self.k = k
        self.num_classes = num_classes

        super(YOLOv2, self).__init__()

        #configuration of the layers in terms of (kernel_size, filter_channel_output). 'M' stands for maxpooling.
        self.cfgs = {
            'yolo':[(3, 32), 'M', (3, 64), 'M', (3, 128), (1, 64), (3, 128), 'M', (3, 256), (1, 128), (3, 256), 'M', (3, 512),
                    (1, 256), (3, 256), (1, 256), (3, 512), 'M', (3, 1024), (1, 512), (3, 1024), (1, 512), (3, 1024), (3, 1024), (3, 1024), (3, 1024),
                    (cfg.DETECTION_CONV_SIZE, self.k*(self.num_classes+5))]
        }

        layers = []
        in_channels = 3


        for lyr in self.cfgs['yolo']:

            padding_value = 1

            if lyr == 'M':
                layers += [NN.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if lyr[0] == 1: #if the filter size is 1, no padding is required. Else, the input dimension will increase by 1.
                    padding_value = 0
                conv2d = NN.Conv2d(in_channels, lyr[1], kernel_size=lyr[0], padding=padding_value)

                #for the last convolution layer. No activation function. No Batch Norm.
                if lyr[1] == k*(num_classes+5):
                    layers += [conv2d]
                    break

                layers += [conv2d, NN.BatchNorm2d(num_features=lyr[1]), NN.LeakyReLU(inplace=True)]
                in_channels = lyr[1]

        self.features = NN.Sequential(*layers)


        if init_weights:
            self._initialize_weights()


    def forward(self, x):

        x = self.features(x)

        subsampled_feature_size = x.size()[-1] #get the size of the feature map

        x = torch.transpose(x, 1, -1)
        #reshape the output into [batch size, feature map width, feature map height, number of anchors, 5 + number of classes]
        x = x.view(-1, subsampled_feature_size, subsampled_feature_size, self.k, 5+self.num_classes)

        x = activate(x)

        return x

def activate(prediction):
    '''
    Apply activation functions on the predicted confidence and centers.
    '''
    prediction[:, :, :, :, 0:3] = NN.Sigmoid()(prediction[:, :, :, :, 0:3])

    return prediction



def loss(predicted_array, label_array):
    '''
    Loss function for Yolov2.
    '''

    #information on which anchor contains object and which anchor don't.
    gt_objectness = label_array[:,:,:,:,0:1]

    mask = torch.ones_like(gt_objectness)

    gt_no_objectness = mask - gt_objectness

    #get the center values from both the arrays.
    predicted_center_x = predicted_array[:, :, :, :, 1:2]
    predicted_center_y = predicted_array[:, :, :, :, 2:3]
    gt_center_x = label_array[:, :, :, :, 1:2]
    gt_center_y = label_array[:, :, :, :,2:3]

    center_loss = cfg.LAMBDA_COORD * torch.sum(gt_objectness * ((predicted_center_x - gt_center_x)**2 + (gt_center_y - predicted_center_y)**2))

    #get the height and width values from both the arrays.
    predicted_width = predicted_array[:, :, :, :, 3:4]
    predicted_height = predicted_array[:, :, :, :, 4:5]
    gt_width = label_array[:, :, :, :, 3:4]
    gt_height = label_array[:, :, :, :, 4:5]

    #actual size loss according to YOLO 1 loss function. But I don't think square root would work since the ground truth values
    #can have negative values.
    # size_loss = cfg.lambda_coord * torch.sum(gt_objectness * ((torch.sqrt(predicted_width + cfg.epsilon_value) - torch.sqrt(gt_width)**2 +
    #                                                 (torch.sqrt(predicted_height + cfg.epsilon_value) - torch.sqrt(gt_height)**2))))

    #mean square size loss without the square roots.
    size_loss = cfg.LAMBDA_COORD * torch.sum(gt_objectness * ((predicted_width  - gt_width)**2 + (predicted_height - gt_height)**2))

    #get the predicted probability of objectness
    predicted_objectness = predicted_array[:, :, :, :, 0:1]

    objectness_loss = torch.sum(gt_objectness*(gt_objectness - predicted_objectness)**2)

    wrong_objectness_loss = cfg.LAMBDA_NOOBJ * torch.sum(gt_no_objectness*(gt_objectness - predicted_objectness)**2)

    #get the predicted probability of the classes and the ground truth class probabilities.
    predicted_classes = gt_objectness * predicted_array[:, :, :, :, 5:] #zeroes the arrays that does not contain object!

    #cross entropy requires the format of (N,C,d1,d2,...dk) where N : batch size, C: class size, and di are other dimensions for the predicted vectors.
    #as for the target, it requires the format of (N, d1,d2,..,dk) and the values are from 0 to number of classes (Scalar values).
    predicted_classes = predicted_classes.contiguous().transpose(1, -1).transpose(-3, -2)
    gt_classes = label_array[:, :, :, :, 5].type(torch.long).contiguous().transpose(1, -1)

    #cross-entropy loss between the predicted and label and sum all the losses.
    classification_loss = NN.CrossEntropyLoss(reduction="mean")(target=gt_classes, input=predicted_classes)

    #sum all the losses together
    total_loss = center_loss + size_loss + objectness_loss + wrong_objectness_loss + classification_loss

    return total_loss





YOLO = YOLOv2(k=cfg.K, num_classes=cfg.NUM_OF_CLASS, init_weights=True)

OPTIMIZER = Adam(YOLO.parameters(), lr=cfg.LEARNING_RATE, weight_decay=5e-5)
LR_DECAY = lr_scheduler.ExponentialLR(OPTIMIZER, gamma=cfg.LEARNING_RATE_DECAY)



if torch.cuda.is_available():

    YOLO = YOLO.cuda()








