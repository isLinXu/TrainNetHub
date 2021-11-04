'''
Testing script for YOLO.
1) Read the file names from the input folder to a list.
2) Load the files using pyTorch dataloader.
3) Feed the dataset by batches into the evaluation model.
4) Use NMS on the output.
5) Draw the bounding box(es) on the images and write the modified images into the output folder.
'''
import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from yolo_net import YOLO
from load_data import ToTensor, LoadTestData
from utils import draw_box
from post_process import PostProcess
import cfg



#load the saved yolo model.
SAVED_MODEL = torch.load(cfg.TRAINED_MODEL_PATH_FOLDER+cfg.TRAINED_MODEL_NAME)
YOLO.load_state_dict(SAVED_MODEL)
YOLO.eval() #yolo evaluation mode.

TEST_DATA = LoadTestData(resized_image_size=cfg.TEST_IMAGE_SIZE, transform=ToTensor(mode='test'))

#check if the anchor sizes are available for the given image size.
try:
    _ = TEST_DATA.all_anchor_sizes[str(cfg.TEST_IMAGE_SIZE)]
except KeyError:
    print("No anchors for the given image size!")
    sys.exit()


DATALOADER = DataLoader(TEST_DATA, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)
POST_PROCESSING = PostProcess(box_num_per_grid=cfg.K, feature_size=cfg.TEST_IMAGE_SIZE//cfg.SUBSAMPLED_RATIO, anchors_list=TEST_DATA.anchors_list)

for i, sample in tqdm(enumerate(DATALOADER)):

    batch_x = sample['image'].cuda()

    outputs = YOLO(batch_x)

    nms_output = POST_PROCESSING.nms(predictions=outputs.detach().clone().contiguous())

    draw_box(image_tensor=batch_x.detach().clone(), pred_tensor=nms_output.detach().clone(),
             classes=cfg.CLASSES.copy(), output_folder=cfg.OUTPUT_FOLDER_PATH, conf_thresh=cfg.CONFIDENCE_THRESH, start=i*cfg.BATCH_SIZE)
