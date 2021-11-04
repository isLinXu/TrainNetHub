'''
ImageNet and YOLO model training.
'''
import dbm
import itertools
import os
from random import randint
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from load_data import LoadDataset, ToTensor, ImgnetLoadDataset
import cfg
from yolo_net import YOLO, OPTIMIZER, LR_DECAY, loss
from post_process import PostProcess
from darknet19 import DARKNET19, IMGNET_OPTIMIZER, IMGNET_LR_DECAY, IMGNET_CRITERION, calculate_accuracy
from utils import calculate_map


#If the classification model is not present, then we'll have to train the model with the ImageNet images for classification.
if not cfg.IMGNET_MODEL_PRESENCE:

    print(DARKNET19)

    IMGNET_TRAINING_DATA = ImgnetLoadDataset(resized_image_size=cfg.IMGNET_IMAGE_SIZE, class_list=cfg.IMGNET_CLASSES,
                                             dataset_folder_path=cfg.IMGNET_DATASET_PATH, transform=ToTensor())

    IMGNET_DATALOADER = DataLoader(IMGNET_TRAINING_DATA, batch_size=cfg.IMGNET_BATCH_SIZE, shuffle=True, num_workers=4)

    BEST_ACCURACY = 0
    for epoch_idx in range(cfg.IMGNET_TOTAL_EPOCH):

        epoch_training_loss = []
        epoch_accuracy = []

        for i, sample in tqdm(enumerate(IMGNET_DATALOADER)):

            batch_x, batch_y = sample["image"].cuda(), sample["label"].type(torch.long).cuda()

            IMGNET_OPTIMIZER.zero_grad()

            classification_output = DARKNET19(batch_x)

            training_loss = IMGNET_CRITERION(input=classification_output, target=batch_y)

            epoch_training_loss.append(training_loss.item())

            training_loss.backward()
            IMGNET_OPTIMIZER.step()

            batch_acc = calculate_accuracy(network_output=classification_output, target=batch_y)
            epoch_accuracy.append(batch_acc.item())


        IMGNET_LR_DECAY.step()

        current_accuracy = np.average(epoch_accuracy)
        print("Epoch %d, \t Training Loss : %g, \t Training Accuracy : %g"%(epoch_idx, np.average(epoch_training_loss), current_accuracy))

        if current_accuracy > BEST_ACCURACY:
            BEST_ACCURACY = current_accuracy
            torch.save(DARKNET19.state_dict(), cfg.IMGNET_MODEL_SAVE_PATH_FOLDER+cfg.IMGNET_MODEL_SAVE_NAME)


#Transfer Learning
IMGNET_MODELLOAD = torch.load(cfg.IMGNET_MODEL_SAVE_PATH_FOLDER+cfg.IMGNET_MODEL_SAVE_NAME)
ALL_KEYS = IMGNET_MODELLOAD.keys()
TOTAL_KEYS = len(ALL_KEYS)

#exclude the last 2 keys (the classification layer's weight and bias)
TRANSFER_LEARNING_PARAMS = dict(itertools.islice(IMGNET_MODELLOAD.items(), TOTAL_KEYS-2))

YOLO.load_state_dict(TRANSFER_LEARNING_PARAMS, strict=False) #load the cnn weights.


#check if a YOLO saved model exists. If yes, load the model.
if os.path.exists(cfg.TRAINED_MODEL_PATH_FOLDER+cfg.TRAINED_MODEL_NAME):

    YOLO_PARAMS = torch.load(cfg.TRAINED_MODEL_PATH_FOLDER+cfg.TRAINED_MODEL_NAME)
    YOLO.load_state_dict(YOLO_PARAMS)
    print("YOLO loaded!")

print(YOLO)

HIGHEST_MAP = 0

TRAINING_LOSSES_LIST = []
TRAINING_MAPS_LIST = []
TRAINING_AP_LIST = []

for epoch_idx in range(cfg.TOTAL_EPOCH):

    epoch_loss = 0
    training_loss = []

    if epoch_idx % 10 == 0:
        #there are 10 options for image sizes.
        chosen_image_index = randint(0, 9)

    #resets the learning rate after every 200 epochs.
    if epoch_idx % 200 == 0 and epoch_idx != 0:
        for g in OPTIMIZER.param_groups:
            g['lr'] = 1e-5


    chosen_image_size = cfg.IMAGE_SIZES[chosen_image_index]
    feature_size = int(chosen_image_size/cfg.SUBSAMPLED_RATIO)

    print("The chosen image size is : ", chosen_image_size)

    training_data = LoadDataset(resized_image_size=chosen_image_size, transform=ToTensor())

    dataloader = DataLoader(training_data, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)

    postProcess_obj = PostProcess(box_num_per_grid=cfg.K, feature_size=feature_size, anchors_list=training_data.anchors_list)

    for i, sample in tqdm(enumerate(dataloader)):

        batch_x, batch_y = sample["image"].cuda(), sample["label"].cuda()

        OPTIMIZER.zero_grad()

        #[batch size, feature map width, feature map height, number of anchors, 5 + number of classes]
        outputs = YOLO(batch_x) #THE OUTPUTS ARE NOT YET GONE THROUGH THE ACTIVATION FUNCTIONS.

        total_loss = loss(predicted_array=outputs, label_array=batch_y)

        #for every 10 epochs, suppress the outputs using Non-max suppression and collect the prediction and ground truth arrays for the calculation
        #of mAP.
        if epoch_idx % 10 == 0:
            #Suppress the prediction outputs.
            nms_output = postProcess_obj.nms(predictions=outputs.detach().clone().contiguous())

            #collect every mini-batch prediction and ground truth arrays.
            postProcess_obj.collect(network_output=nms_output.clone(), ground_truth=batch_y.clone())

        training_loss.append(total_loss.item())
        total_loss.backward()
        OPTIMIZER.step()



    LR_DECAY.step() #decay rate update

    training_loss = np.average(training_loss)
    print("Epoch %d, \t Loss : %g"%(epoch_idx, training_loss))

    TRAINING_LOSSES_LIST.append(training_loss)

    #calculate mean average precision for every 10 epochs
    if epoch_idx % 10 == 0:

        avg_prec = postProcess_obj.calculate_ap()
        print(avg_prec)
        mean_ap = calculate_map(avg_prec)
        postProcess_obj.clear_lists() #clears the list after every mAP calculation.
        print("Mean AP : ", mean_ap)
        TRAINING_AP_LIST.append(avg_prec)
        TRAINING_MAPS_LIST.append(mean_ap)
        if mean_ap > HIGHEST_MAP: #save the model with the highest mAP.
            HIGHEST_MAP = mean_ap
            torch.save(YOLO.state_dict(), cfg.TRAINED_MODEL_PATH_FOLDER+cfg.TRAINED_MODEL_NAME)


        AP_FILE = open("ap.txt", 'w')
        AP_FILE.write(str(TRAINING_AP_LIST))
        AP_FILE.close()


        MAP_FILE = open("map.txt", 'w')
        MAP_FILE.write(str(TRAINING_MAPS_LIST))
        MAP_FILE.close()

    LOSS_FILE = open("loss.txt", "w")
    LOSS_FILE.write(str(TRAINING_LOSSES_LIST))
    LOSS_FILE.close()

    #write the anchor sizes data on the file from the database.
    YOLO_DB = dbm.open(cfg.YOLO_DB, 'c')
    ANCHOR_FILE = open(cfg.ANCHOR_BOXES_STORE, 'w')
    ANCHOR_DATA = [x for x in YOLO_DB.items()]
    ANCHOR_FILE.write(str(ANCHOR_DATA))
    ANCHOR_FILE.close()
