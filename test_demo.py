import torch
import torchvision.models as models
# import Vision.models as models

from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet18 = models.resnet18()
alexnet = models.alexnet().to(device)
vgg16 = models.vgg16()
squeezenet = models.squeezenet1_0()
densenet = models.densenet161()
inception = models.inception_v3()
googlenet = models.googlenet()
shufflenet = models.shufflenet_v2_x1_0()
mobilenet = models.mobilenet_v2()
resnext50_32x4d = models.resnext50_32x4d()
wide_resnet50_2 = models.wide_resnet50_2()
mnasnet = models.mnasnet1_0()



print(summary(alexnet, (3, 224, 224)))