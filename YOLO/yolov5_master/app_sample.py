import cv2
import torch
from PIL import Image
from YOLO.yolov5_master.main import PackageProjectUtil

model = torch.hub.load(PackageProjectUtil.project_root_path(), 'custom', path='weights/best.pt', force_reload=True,source = 'local')

img1 = Image.open('/media/hxzh02/SB@home/hxzh/MyGithub/TrainNetHub/YOLO/yolov5_master/data/images/bus.jpg')  # PIL image
img2 = cv2.imread('/media/hxzh02/SB@home/hxzh/MyGithub/TrainNetHub/YOLO/yolov5_master/data/images/zidane.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)
imgs = [img1, img2]  # batch of images

# Inference
results = model(imgs, size=640)  # includes NMS

# Results
results.print()
results.save()  # or .show()
