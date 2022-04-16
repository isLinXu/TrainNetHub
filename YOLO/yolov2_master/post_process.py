'''
Non-Max Suppression and mAP calculation.

1) NMS Process:

Input : Prediction tensor, pred shaped [batch_size, xgrids(feature_size), ygrids(feature_size), anchor_nums_in_a_box(k), 5+class_num]
Output: Tensor shaped [batch_size, total_anchor_nums, 5+class_num] where the suppressed arrays will have values of 0 and the remaining
        arrays will be converted to the box coordinates instead of regression values as the input arrays.

    i)    If pred[:,:,:,:,0] < confidence_threshold : #if the confidence score of an array is lower than the threshold
            pred[:,:,:,i] = 0                         #zero-out the entire array.
    ii)   Convert ENTIRE pred regression values into box coordinates.
    iii)  Reshape pred -> [batch_size, total_anchor_nums(xgrids*ygrids*anchor_nums_in_a_box), 5+class_num]
    iv)  Sort pred based on the confidence scores in the descending order.
    v)   FOR j from 0 to total_anchor_nums DO
             ref_box = pred[:, j]
             check IoU between ref_box and all other prediction arrays, pred[:, j+1:]
             IF the IoU between the ref box and the compared_box is > IoU_Threshold AND class of ref_box & class of compared_box are SAME :
                compared_box = 0


2) mAP process :
    Input : Prediction and ground truth tensors.
    Output: AP for every class.

    1) Rank the predicted boxes in decreasing order based on the confidence score and choose top-N boxes.
        2) For every chosen boxes, determine if they are TP, FP or FN based on its IoU with the ground-truth boxes (There is no TN since an image
        contains at least 1 object).
            a) If the chosen box has more than a certain set iou threshold with a ground-truth box and the class is also predicted correctly, it's a TP.
            b) If the chosen box has more than a certain set iou threshold with a ground-truth box but the class is predicted wrongly, it's a FN.
            c) If a ground-truth box is missed without a predicted box, then it's a FN. (FN is not needed to calculate Interpolated Precision and/or
            Average Precision)
            d) If the chosen box has less than a certain set iou threshold with a ground-truth box but class predicted correctly, it's a FP.
            e) If the chosen box has more than a certain set iou threshold with a ground-truth box, class is predicted correctly, but it's a duplicated
            prediction, then it's a FP.
        3) Calculate the recall and precision based on the TP, FP and FN for each class.
        4) Calculate the Interpolated Precision (Calculated at each recall level) for each class.
        5) Calculate Average Precision (AP) by taking the Area Under Curve of Interpolated Precision for each class.
        6) Mean the AP over all classes.
        NOTE : Precision = TP/all detections ..... Recall = TP/all ground truths

'''

import numpy as np
import torch
import cfg
from utils import nms_iou_check, map_iou_check

class PostProcess:
    '''
    Contains modules to perform non-max suppressions and calculation of mAP.
    '''

    def __init__(self, box_num_per_grid, feature_size, anchors_list, iou_thresh=cfg.MAP_IOU_THRESH, confidence_thresh=cfg.CONFIDENCE_THRESH,
                 subsampled_ratio=cfg.SUBSAMPLED_RATIO, num_class=cfg.NUM_OF_CLASS, nms_iou_thresh=cfg.NMS_IOU_THRESH):
        '''
        Initialize parameters.
        '''

        self.num_of_anchor = box_num_per_grid
        self.anchors_list = torch.from_numpy(anchors_list).to(cfg.DEVICE)
        self.feature_size = feature_size
        self.subsampled_ratio = subsampled_ratio
        self.iou_thresh = iou_thresh
        self.confidence_thresh = confidence_thresh
        self.nms_iou_thres = nms_iou_thresh
        self.epoch_predboxes = []
        self.epoch_gtboxes = []
        self.num_class = num_class

    def collect(self, network_output, ground_truth, suppressed=True):
        '''
        Collects all the prediction arrays and the corresponding ground truth arrays for every image.
        '''

        batch_size = ground_truth.size()[0]
        #if the output is not yet non-max suppressed, then it would have to be converted from the regression values to box coordinates and
        #the array would be reshaped to [batch_size, num_predictions, 5] where the last index would be store the class index.
        if not suppressed:
            network_output = self.get_box_coordinates(in_array=network_output)
            network_output = network_output.view(network_output.size()[0], -1, 5+self.num_class)
            network_output[:, :, 5] = torch.argmax(network_output[:, :, 5:], dim=-1)
            network_output = network_output[:, :, :6]

        #sort the predictions in decreasing order based on the confidence score.
        sorted_index = torch.argsort(input=network_output[:, :, 0], descending=True)
        network_output = network_output[torch.arange(start=0, end=batch_size).view(-1, 1), sorted_index[:, :]]

        ground_truth = self.get_box_coordinates(in_array=ground_truth)
        ground_truth = ground_truth.view(batch_size, -1, 6)

        sorted_index = torch.argsort(input=ground_truth[:, :, 0], descending=True)
        ground_truth = ground_truth[torch.arange(start=0, end=batch_size).view(-1, 1), sorted_index[:, :]]




        #append the transformed prediction array and the ground truth arrays into the lists for mini-batches until the epoch is completed.
        for box in network_output:
            self.epoch_predboxes.append(box.cpu().numpy())

        for box in ground_truth:
            #add another 0 at the end of the ground truth prediction array to check if a gt box is already assignned to a pred box or not.
            placeholder = np.zeros((ground_truth.size()[1], 7))
            placeholder[:, :6] = box.cpu().numpy()
            self.epoch_gtboxes.append(placeholder)


    def clear_lists(self):
        '''
        Empties the lists.
        '''
        self.epoch_predboxes = []
        self.epoch_gtboxes = []


    def calculate_ap(self):
        '''
        Calculates the AP for every class.
        '''
        avg_precision = {}
        self.epoch_gtboxes = np.asarray(self.epoch_gtboxes, dtype=np.float32)
        self.epoch_predboxes = np.asarray(self.epoch_predboxes, dtype=np.float32)

        for class_index in range(self.num_class):

            #recall must always start from 0.
            precision_list, recall_list = [], [0]
            true_positive, false_positive, false_negatives = 0, 0, 0

            #this would be a 2-tuple where the first index contains an array of batch indices and the corresponding pred indices on the second index.
            #we need the second condition as well because there are many zero arrays (no predictions values) as well.
            class_preds_indices = np.where((self.epoch_predboxes[:, :, 5] == class_index) & (self.epoch_predboxes[:, :, 0] > self.confidence_thresh))

            total_class_det = class_preds_indices[0].shape[0]

            for batch_index, pred_index in zip(class_preds_indices[0], class_preds_indices[1]):

                #check for ground truth box(es) on the specified batch index where the class is equal to the class that we're working on. and the
                #confidence is 1. The confidence condition is necessary as there are many zero arrays that would be misidentified as class 0.
                class_gt_indices = np.where((self.epoch_gtboxes[batch_index, :, 5] == class_index) & (self.epoch_gtboxes[batch_index, :, 0] == 1))


                num_gt_box_class = class_gt_indices[0].shape[0] #total number of ground truth boxes that satisfies our search condition above.

                #if there are no ground-truth boxes found that belongs to the same class and are unassigned, continue the loop and FP += 1.
                if num_gt_box_class == 0:

                    false_positive += 1

                    precision = true_positive/total_class_det

                    try:
                        recall = true_positive/num_gt_box_class
                    except ZeroDivisionError:
                        recall = 0

                    precision_list.append(precision)
                    recall_list.append(recall)

                    continue

                ref_box = self.epoch_predboxes[batch_index, pred_index] #the prediction box to be compared (iou) with the gt box(es).

                for gt_index in class_gt_indices[0]: #iterate through all the detected ground truth boxes.

                    gt_box = self.epoch_gtboxes[batch_index, gt_index]

                    iou = map_iou_check(box_a=ref_box, box_b=gt_box)

                    if iou > self.iou_thresh and self.epoch_gtboxes[batch_index, gt_index, -1] == 0:

                        #assign the ground truth box
                        self.epoch_gtboxes[batch_index, gt_index, -1] = 1.0
                        true_positive += 1

                        precision = true_positive/total_class_det

                        try:
                            recall = true_positive/num_gt_box_class
                        except ZeroDivisionError:
                            recall = 0

                        precision_list.append(precision)
                        recall_list.append(recall)


                    else:
                        false_positive += 1

                        precision = true_positive/total_class_det

                        try:
                            recall = true_positive/num_gt_box_class
                        except ZeroDivisionError:
                            recall = 0

                        precision_list.append(precision)
                        recall_list.append(recall)

            #check for all the unassigned ground truth boxes for the calculation of false negatives.
            unassigned_gtbox_indices = np.where((self.epoch_gtboxes[:, :, 5] == class_index) & (self.epoch_gtboxes[:, :, 0] == 1)
                                                & (self.epoch_gtboxes[:, :, -1] == 0))

            false_negatives += unassigned_gtbox_indices[0].shape[0]

            #Calculate IP.
            #The array is reversed before we accumulate in order to go from high value to low value.
            #If the array is not reversed, then if the first detection is correct, i.e. precision = 1, then the rest of the array will be 1.
            reversed_precision_list = np.asarray(precision_list[::-1])
            interpolated_precision = np.flip(np.maximum.accumulate(reversed_precision_list))

            area_under_curve = 0

            #calculate the area under curve.
            for recall_index in range(len(recall_list) - 1):

                curr_recall = recall_list[recall_index]
                next_recall = recall_list[recall_index + 1]

                 #+1 on the ip index is not needed as the recall basically starts one step behind since we added 0 to its list in the initialization.
                chosen_ip = interpolated_precision[recall_index]

                #sometimes the recall value might go down due to the loss of correct prediction. In those times, (next_recall - curr_recall) becomes
                #less than 0. Area under curve should never be < 0. Hence this error handling.
                current_area = max(0, (next_recall - curr_recall)*chosen_ip)

                area_under_curve += current_area

                #stop when recall reaches 1. Else, there might be times when recall decreases and increases and this will contribute to AP
                #value to be more than 100.
                if int(next_recall) == 1:
                    break

            avg_precision[cfg.CLASSES[class_index]] = area_under_curve

        return avg_precision #returns the average precision for each class in a dictionary.


    def get_box_coordinates(self, in_array):
        '''
        Given the regression values from the prediction of the network, calculate back the predicted box's coordinate for the entire batch.
        Can also be used as a standalone module apart from nms related.
        '''

        #set the array values for the predicted probability lower than the threshold to 0
        in_array[in_array[:, :, :, :, 0] < self.confidence_thresh] = 0

        #contains the grid indexes of the values in the pred array in the shape of [num_of_dimension(5), batch_size, xgrids, ygrids, anchor_index,25]
        #each dimension(axis 0) contains the index num of the following dimensions
        #[0] contains the batch index, [1] contains the xgrids, ... [4] contains the 25 length vectors.
        all_indices = torch.from_numpy(np.indices(dimensions=in_array.size())).to(cfg.DEVICE)

        all_x = in_array[:, :, :, :, 1:2] #all the x-coor regression values
        all_y = in_array[:, :, :, :, 2:3]
        all_w = in_array[:, :, :, :, 3:4]
        all_h = in_array[:, :, :, :, 4:5]

        anchors_w = self.anchors_list[:, :, :, 3:4]
        anchors_h = self.anchors_list[:, :, :, 4:5]

        grid_x = all_indices[1][:, :, :, :, 1:2]
        grid_y = all_indices[2][:, :, :, :, 2:3]

        cvt_w = anchors_w*(torch.exp(all_w))
        cvt_h = anchors_h*(torch.exp(all_h))
        cvt_x = (all_x*self.subsampled_ratio) + (grid_x*self.subsampled_ratio)
        cvt_y = (all_y*self.subsampled_ratio) + (grid_y*self.subsampled_ratio)

        #calculate the x1,y1,x2,y2 and insert them in the index 1,2,3 and 4 in the prediction array.
        in_array[:, :, :, :, 1:2] = cvt_x - cvt_w/2
        in_array[:, :, :, :, 2:3] = cvt_y - cvt_h/2
        in_array[:, :, :, :, 3:4] = cvt_x + cvt_w/2
        in_array[:, :, :, :, 4:5] = cvt_y + cvt_y/2

        #replace center x,y and w,h with x1,y1,x2,y2
        return in_array

    def nms(self, predictions):
        '''
        Perform Non-Max Suppression and returns the prediction arrays in suppressed form.
        '''

        batch_size = predictions.size()[0]
        cvt_arrays = self.get_box_coordinates(in_array=predictions)

        num_predictions = self.feature_size*self.feature_size*self.num_of_anchor

        #THE PREDICTION ARRAY IS TO BE RESHAPED TO [BATCH_SIZE, TOTAL_PREDICTION_BOXES, 5+NUM_CLASS]. ONCE THE RESHAPING IS DONE, THE ARRAY
        #WILL BE THEN SORTED ACCORDING TO THE CONFIDENCE VALUES WHICH IS LOCATED IN [:,:,0] THE FIRST INDEX IN THE THIRD AXIS.
        reshaped_pred = cvt_arrays.view(-1, num_predictions, 5+self.num_class)


        #convert the one-hot vector of the class label into the index of the class.
        reshaped_pred[:, :, 5] = torch.argmax(reshaped_pred[:, :, 5:], dim=-1)
        reshaped_pred = reshaped_pred[:, :, :6]



        sorted_index = torch.argsort(input=reshaped_pred[:, :, 0], descending=True) #sorting on the third axis, the location of the confidence scores.
        #We cannot simply insert the sorted_index variable into the reshaped_pred as this is a 3D array, not 2D. Simply inserting the indexes
        #will throw an error stating that the value is out of bound for axis 0 with size [BATCH_SIZE]. This is due to the fact that
        #the sorted_index contains the value of from 0 to num_predictions in the shape of [batch_size,num_predictions]. We cannot insert it with
        #reshaped_pred[:, sorted_index] either as it will create an extra dimension with the size of the [BATCH_SIZE].
        # Therefore, we need to specify the batch indices as well. Hence the torch.arange.
        sorted_pred = reshaped_pred[torch.arange(start=0, end=batch_size).view(-1, 1), sorted_index[:, :]]


        #zero-out the arrays that contains lower than the confidence threshold.
        boolean_array = sorted_pred[:, :, 0] < self.confidence_thresh
        sorted_pred[boolean_array] = 0

        #we will iterate through every prediction in all the batches and suppress the boxes that belongs to the same class of another box
        #that has a higher confidence and the IoU between them is more than the set threshold.
        for i in range(num_predictions-1): #-1 since the last prediction has nothing to be compared to.

            ref_pred = sorted_pred[:, i:i+1].clone() # i+1 is to retain the dimensions.
            ref_class = sorted_pred[:, i:i+1, 5:6].clone()

            comparing_arrays = sorted_pred[:, i+1:].clone() #make a copy of the arrays that we'll be comparing our reference pred to.

            #whichever prediction that does not belong to the same class as the reference pred array will be zeroed out.
            comparing_arrays = torch.where(comparing_arrays[:, :, 5:6] == ref_class, comparing_arrays, torch.Tensor([0.]).to(cfg.DEVICE))

            #get the iou between the reference pred array and all other remaining arrays on the right.
            iou_batch = nms_iou_check(box_a=ref_pred, box_b=comparing_arrays, device=cfg.DEVICE)

            #NOTE that we're not using comparing_arrays in the torch.where because the arrays that do not have the same class
            #with the reference array were zeroed. Whichever array that has the iou more than the threshold, the confidence value
            #will be zeroed.
            sorted_pred[:, i+1:, 0] = torch.where((iou_batch < self.nms_iou_thres), sorted_pred[:, i+1:, 0], torch.Tensor([0.]).to(cfg.DEVICE))

        #zero the entire array that has confidence lower than the threshold again.
        boolean_array = sorted_pred[:, :, 0] < self.confidence_thresh
        sorted_pred[boolean_array] = 0

        return sorted_pred
