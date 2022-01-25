import json
from PIL import Image
import torch
from torchvision import transforms

from Efficient.EfficientNet_master.efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b0')
# model = EfficientNet.from_pretrained('efficientnet-b0.pth')
net_name = 'efficientnet-b0'
model = EfficientNet.from_name(net_name)
net_weight = 'eff_weights/' + 'efficientnet-b0-355c32eb.pth'
state_dict = torch.load(net_weight)
model.load_state_dict(state_dict)


# Preprocess image
tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
path = '/media/hxzh02/SB@home/hxzh/MyGithub/TrainNetHub/Efficient/EfficientNet_master/Flowers/test/1/4.jpg'
img = tfms(Image.open(path)).unsqueeze(0)
print(img.shape) # torch.Size([1, 3, 224, 224])

# Load ImageNet class names
labels_map = json.load(open('labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify
model.eval()
with torch.no_grad():
    outputs = model(img)

# Print predictions
print('-----')
for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))