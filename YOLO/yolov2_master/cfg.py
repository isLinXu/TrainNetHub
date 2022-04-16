'''
Configuration file.
'''
import os
import torch
from utils import imgnet_get_classes, imgnet_check_model, create_training_lists, create_test_lists


###IMAGENET config
IMGNET_DATASET_PATH = '../ImageNet'
IMGNET_CLASSES = imgnet_get_classes(folder_path=IMGNET_DATASET_PATH)
IMGNET_NUM_OF_CLASS = len(IMGNET_CLASSES)
IMGNET_MODEL_SAVE_PATH_FOLDER = './imagenet_model/'
IMGNET_MODEL_SAVE_NAME = 'imagenet_model.pth'
IMGNET_MODEL_PRESENCE = imgnet_check_model(model_path=IMGNET_MODEL_SAVE_PATH_FOLDER+IMGNET_MODEL_SAVE_NAME)
IMGNET_LEARNING_RATE = 1e-3
IMGNET_LEARNING_RATE_DECAY = 0.9
IMGNET_TOTAL_EPOCH = 160
IMGNET_BATCH_SIZE = 50
IMGNET_IMAGE_SIZE = 224
###


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_IMAGES_PATH = '../VOCdevkit/VOC2012/JPEGImages'
DATA_ANNOTATION_PATH = '../VOCdevkit/VOC2012/Annotations'
TRAINED_MODEL_PATH_FOLDER = './yolo_model/'
TRAINED_MODEL_NAME = 'yolo.pth'
TEST_FOLDER_PATH = './test_images/'
OUTPUT_FOLDER_PATH = './output/'
ANCHOR_BOXES_STORE = 'anchor_sizes.txt'
YOLO_DB = 'yolo_db'
IMAGE_SIZES = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
TEST_IMAGE_SIZE = 448
IMAGE_DEPTH = 3
DETECTION_CONV_SIZE = 3
SUBSAMPLED_RATIO = 32
K = 5 #number of anchor box in a grid
LEARNING_RATE = 1e-5
LEARNING_RATE_DECAY = 0.999
LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5
TOTAL_EPOCH = 1000
MAP_IOU_THRESH = 0.5
CONFIDENCE_THRESH = 0.8
BATCH_SIZE = 5
NMS_IOU_THRESH = 0.75
EXCLUDED_CLASSES = ['horse', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'chair', 'cow', 'diningtable', 'person','cat',
                    'motorbike', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']



#Get the image and annotation file paths
#resized image size does not affect anything since we're just using it to filter the unwanted classes.
LIST_IMAGES, LIST_ANNOTATIONS, ALL_CLASSES = create_training_lists(data_images_path=DATA_IMAGES_PATH, data_annotation_path=DATA_ANNOTATION_PATH,
                                                                   excluded_classes=EXCLUDED_CLASSES, resized_image_size=IMAGE_SIZES[0])

TEST_IMAGE_LIST = create_test_lists(TEST_FOLDER_PATH)

#get the classes for all the training data.
CLASSES = sorted([x for x in ALL_CLASSES if not x in EXCLUDED_CLASSES])


TOTAL_IMAGES = len(LIST_IMAGES)
NUM_OF_CLASS = len(CLASSES)

#create the directories if they don't exist.
if not os.path.exists(IMGNET_MODEL_SAVE_PATH_FOLDER):
    os.makedirs(IMGNET_MODEL_SAVE_PATH_FOLDER)

if not os.path.exists(TRAINED_MODEL_PATH_FOLDER):
    os.makedirs(TRAINED_MODEL_PATH_FOLDER)

if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)

if not os.path.exists(TEST_FOLDER_PATH):
    os.makedirs(TEST_FOLDER_PATH)

if not os.path.exists(ANCHOR_BOXES_STORE):
    os.mknod(ANCHOR_BOXES_STORE)
