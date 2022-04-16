'''
Was used to test YOLO's correctness.
'''

'''
from load_data import LoadDataset, ToTensor
import cfg
import torch
from torch.utils.data import DataLoader
import cv2
from label_format import calculate_ground_truth
import numpy as np
from yolo_net import YOLO, OPTIMIZER, loss, LR_DECAY #decay rate update
from tqdm import tqdm
from post_process import PostProcess
from random import randint
# from darknet19 import darknet19, ImgNet_optimizer, ImgNet_lr_decay, ImgNet_criterion
import itertools
import os
from utils import calculate_map


chosen_image_index = 0
highest_map = 0

training_losses_list = []
training_mAPs_list = []

if os.path.exists(cfg.TRAINED_MODEL_PATH_FOLDER+cfg.TRAINED_MODEL_NAME):

    YOLO_PARAMS = torch.load(cfg.TRAINED_MODEL_PATH_FOLDER+cfg.TRAINED_MODEL_NAME)
    YOLO.load_state_dict(YOLO_PARAMS)
    print("YOLO loaded!")


YOLO = YOLO.eval()
print(YOLO)

training_data = LoadDataset(resized_image_size=320, transform=ToTensor())

dataloader = DataLoader(training_data, batch_size=1, shuffle=False, num_workers=4)
for epoch_idx in range(cfg.TOTAL_EPOCH):

    epoch_loss = 0
    training_loss = []



    chosen_image_size = 320
    feature_size = int(chosen_image_size/cfg.SUBSAMPLED_RATIO)



    postProcess_obj = PostProcess(box_num_per_grid=cfg.K, feature_size=feature_size, anchors_list=training_data.anchors_list)


    for i, sample in tqdm(enumerate(dataloader)):
        # print(sample["image"].shape)
        # print(sample["label"].shape)
        lear_rate = None
        for g in OPTIMIZER.param_groups:
            lear_rate = g['lr']

        print("Epoch %d, LR : %g"%(epoch_idx, lear_rate))

        if i == 1:
            pass
        else:
            continue

        batch_x, batch_y = sample["image"].to(cfg.DEVICE), sample["label"].to(cfg.DEVICE)

        OPTIMIZER.zero_grad()

        #[batch size, feature map width, feature map height, number of anchors, 5 + number of classes]
        outputs = YOLO(batch_x) #THE OUTPUTS ARE NOT YET GONE THROUGH THE ACTIVATION FUNCTIONS.


        total_loss = loss(predicted_array=outputs, label_array=batch_y)
        # mAP_object._collect(predicted_boxes=outputs.detach().cpu().numpy(), gt_boxes=batch_y.cpu().numpy())
        # mAP_object.non_max_suppression(predictions=outputs.detach().cpu().numpy())
        nms_output = postProcess_obj.nms(predictions=outputs.detach().clone().contiguous())
        postProcess_obj.collect(network_output=nms_output.clone(), ground_truth=batch_y.clone())

        training_loss.append(total_loss.item())
        total_loss.backward()
        OPTIMIZER.step()
        img_ = np.asarray(np.transpose(batch_x.cpu().numpy()[0], (1, 2, 0)))
        img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

        # calculated_batch = calculate_ground_truth(subsampled_ratio=32, anchors_list=training_data.anchors_list, resized_image_size=chosen_image_size,
                            # network_prediction=outputs.detach().cpu().numpy(), prob_threshold=0.7)
        print(nms_output.shape)
        np_output = nms_output.contiguous().cpu().numpy()
        gt_box = calculate_ground_truth(subsampled_ratio=32, anchors_list=training_data.anchors_list,
                            network_prediction=batch_y.cpu().numpy(), prob_threshold=0.9, ground_truth_mode=True)
        # print("CLASS : " , calculated_batch[0])

        for k in range(gt_box.shape[1]):
            # print(int(calculated_batch[0][k][0]), int(calculated_batch[0][k][1]), int(calculated_batch[0][k][2]), int(calculated_batch[0][k][3]))
            # try:

            cv2.putText(img, (str(cfg.CLASSES[int(gt_box[0][k][5])])), (int(gt_box[0][k][1])+20,
                                                                                  int(gt_box[0][k][2])-10), cv2.FONT_HERSHEY_SIMPLEX,
                                                                                                                                    0.4, (36,255,12), 2)
            cv2.rectangle(img, (int(gt_box[0][k][1]), int(gt_box[0][k][2])), (int(gt_box[0][k][3]), int(gt_box[0][k][4])),
                        (255,0,0), 1)
            # except Exception as e:
            #     print(e)
            #     pass
        for k in range(np_output.shape[1]):
            # print(int(calculated_batch[0][k][0]), int(calculated_batch[0][k][1]), int(calculated_batch[0][k][2]), int(calculated_batch[0][k][3]))
            try:

                cv2.putText(img, (str(round(np_output[0][k][0],4))+", "+ str(cfg.CLASSES[int(np_output[0][k][5])])), (int(np_output[0][k][1]),
                                                                                    int(np_output[0][k][2])-8), cv2.FONT_HERSHEY_SIMPLEX,
                                                                                                                                        0.4, (36,255,12), 2)
                cv2.rectangle(img, (int(np_output[0][k][1]), int(np_output[0][k][2])), (int(np_output[0][k][3]), int(np_output[0][k][4])),
                        (0, 255, 0), 1)
            except:
                pass

        cv2.imshow("img", img)
        cv2.waitKey(0)
        break

    avg_prec = postProcess_obj.calculate_ap()
    print(avg_prec)
    mean_ap = calculate_map(avg_prec)
    print(mean_ap)
    postProcess_obj.clear_lists()

    LR_DECAY.step() #decay rate update

    # meanAP = mAP_object.calculate_meanAP()
    # print("MEAN Avg Prec : ", meanAP)
    training_loss = np.average(training_loss)
    print("Epoch %d, \t Loss : %g"%(epoch_idx, training_loss))

    training_losses_list.append(training_loss)
    # training_mAPs_list.append(meanAP)
'''
