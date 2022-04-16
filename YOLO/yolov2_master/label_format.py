'''
YOLOv2 requires the label for each image to be in a particular format.
Anchors are placed consistently all over the image based on the subsampled size (the final feature map size). E.g. a 320 x 320 image with a
subsampling ratio of 32 will have a feature map of size 10 x 10. Anchors are placed on each of this 10 x 10 grid but the center coordinates, width and
height of the anchors are referenced to the original image. Therefore, the center or anchors are placed with a gap of 32 pixels horizontally and
vertically on the original 320 x 320 image. We need to identify which grid (on the feature map) does the ground-truth bounding box's center falls into.
Then we compare the IoU between the ground-truth box and every anchor that belongs to the particular grid. Whichever anchor that has the highest IoU
with the ground-truth box will be responsible for detecting the object. NOTE: When we visualize the grids on the feature map, the anchors are located
on the intersection of the grids, NOT inside the grid.

Each anchor has 5 elements. Probability of objectness, center x, center y, width and height. The neural network will be forced to learn the probability
of objectness on each anchor and if the probability is above a certain threshold, the neural network will predict the offset of the anchor from
the ground-truth bounding box. However, the offset is not as simple as calculating the difference between the anchor box and the ground-truth box.
In order to maintain the model's stability, we will normalize both the anchor box and the ground-truth box based on the grid-cell's width and height so
that the center coordinates and height & width of both of the boxes falls in the range of 0 and 1. From there, we calculate the difference between
the value the model should predict (tx, ty) and the anchor box's center values to get the bounding box's center values. For the width and height,
refer to the YOLOv2 paper for the formula. Since the objectness probablity and center coordinates offset that the model should predict is from 0 to 1,
we will be using sigmoid activation function for them. However, the offset values for width and height can be a value from -1 to 1 (maybe TanH ?
The paper does not state the activation function that should be used for this).
'''

import math
import numpy as np

def get_highest_iou_anchor(anchors, gt_box):
    '''
    Calculates the IoU between all the anchors and one ground-truth box.
    Returns the index of the anchor with the highest IoU provided the anchor has not been assigned with an object already. Else the next anchor
    with the highest IoU will be returned.
    '''
    #extract the height and widths of the boxes.
    heights = np.minimum(anchors[:, -2], gt_box[2])
    widths = np.minimum(anchors[:, -1], gt_box[3])

    if np.count_nonzero(heights == 0) > 0 or np.count_nonzero(widths == 0) > 0:
        raise ValueError('The given box has no area!')

    #since the height and width of the boxes have the same origin, we can calculate the intersection simply by multiplying them.
    intersection_area = heights*widths
    gt_box_area = gt_box[2] * gt_box[3]
    anchors_area = anchors[:, -2] * anchors[:, -1]

    ious = intersection_area/ (gt_box_area + anchors_area - intersection_area)

    highest_ious = np.argpartition(ious, -1) #get the sorted indexes.

    #we need to find the anchor with the next highest IoU if the current found anchor is already associated with a ground-truth box.
    index = 0
    current_highest = highest_ious[index]
    while int(anchors[current_highest][0]) != 0: #iterate as long as the selected anchor is occupied.
        index += 1
        current_highest = highest_ious[index] #choose the next highest IoU anchor.

        #in the event of all anchors being occupied, the first anchor with the highest IoU will be chosen.
        if index == len(anchors): #stop the loop once all anchors are considered.
            current_highest = highest_ious[0]
            break

    return current_highest




def label_formatting(gt_class_labels, gt_boxes, anchors_list, subsampled_ratio, resized_image_size):
    '''
    Formats the given labels from the xml file into YOLO's format as explained above.
    '''

    subsampled_size = int(resized_image_size/subsampled_ratio)

    #this array will be used to store the ground truth probability of objectness,  offset calculations between the responsible anchors
    #and the ground-truth boxes and the class of the object. The class of the object will just be an integer since PyTorch's cross entropy
    #will convert it into one hot label for us.
    label_array = np.zeros((subsampled_size, subsampled_size, anchors_list.shape[-2], 6), dtype=np.float32)



    #An image can contain more than 1 objects.
    for i in range(gt_boxes.shape[0]):

        class_label_index = gt_class_labels[i] #extract the class index

        #extract the coordinate of the ground truth box. This ground-truth box's format is [x1,y1,x2,y2]

        gt_box_x1 = gt_boxes[i][0]
        gt_box_y1 = gt_boxes[i][1]
        gt_box_x2 = gt_boxes[i][2]
        gt_box_y2 = gt_boxes[i][3]

        #transform the ground truth values to [x,y,w,h] (center coordinates, width and height)
        gt_box_height = gt_box_y2 - gt_box_y1
        gt_box_width = gt_box_x2 - gt_box_x1
        gt_center_x = gt_box_x1 + (gt_box_width/2)
        gt_center_y = gt_box_y1 + (gt_box_height/2)

        gt_box = [gt_center_x, gt_center_y, gt_box_width, gt_box_height]

        #identify the grid that holds the center of the ground truth box in the subsampled feature image.
        responsible_grid = [int(gt_center_x/subsampled_ratio), int(gt_center_y/subsampled_ratio)]

        #these are the anchors from the responsible grid that we calculate the IoU with.
        prospect_anchors = anchors_list[responsible_grid[0]][responsible_grid[1]]

        #get the index of the anchor from the responsible grid with the highest IoU.
        chosen_anchor_index = get_highest_iou_anchor(anchors=prospect_anchors, gt_box=gt_box)

        chosen_anchor = prospect_anchors[chosen_anchor_index] #the chosen anchor. [pr(obj), x, y, w, h]

        #CALCULATION FOR THE REGRESSION VALUES!
        #All anchors' center are located on the intersections of the grids. Therefore, we can calculate the offset of the ground-truth bounding box
        #using the responsible grid's location. Refer to the paper for more information.

        #as for the centers, the network must predict a value such that when the value is added to the responsible grid's coordinate, we get the
        #normalized value of the ground truth box's center.
        sigmoid_tx = gt_center_x/subsampled_ratio - responsible_grid[0]
        sigmoid_ty = gt_center_y/subsampled_ratio - responsible_grid[1]

        #as for the width and height, the network must predict the exponential value to the euler's number multiplied with anchor box's
        #width and height.
        t_w = math.log(gt_box_width/chosen_anchor[3])
        t_h = math.log(gt_box_height/chosen_anchor[4])

        #objectness probability + regression values + class index
        label_values = np.asarray([1.0, sigmoid_tx, sigmoid_ty, t_w, t_h, class_label_index], dtype=np.float32)

        #We need to occupy the probability of objnectness and the regression values in the chosen anchor's index.
        label_array[responsible_grid[0]][responsible_grid[1]][chosen_anchor_index][:] = label_values


    return label_array


def calculate_ground_truth(subsampled_ratio, anchors_list, network_prediction, prob_threshold, ground_truth_mode=False):
    '''
    Given the regression predictions from the network in batches, calculate back the predicted box's coordinates for every image.
    '''

    assert prob_threshold <= 1 and prob_threshold >= 0, "The objectness probability threshold has to be a value from 0 to 1."

    predicted_data_num = network_prediction.shape[0] #number of data in the batch.

    #set the array values for the predicted objectness probability lower than the threshold to 0.
    boolean_array = network_prediction[:, :, :, :, 0] < prob_threshold
    network_prediction[boolean_array] = 0

    entire_batch_transformed_values = []
    entire_anchor = []

    for i in range(predicted_data_num): #loop through every item in the batch.

        predicted_arrays = network_prediction[i] #get the predicted objectness probabilities and regression values for the particular image.

        occupied_array_indexes = np.nonzero(predicted_arrays[:,:,:,0]) #get the indexes of the arrays that does not have 0 as their objectness probability.

        num_of_occupied_arrays = occupied_array_indexes[0].shape[0]

        transformed_values = [] #to hold the transformed values (from predicted regressions into ground truth boxes)
        anchor_values = []

        for j in range(num_of_occupied_arrays): #loop through every occupied anchors in a particular batch index.

            grid_x = occupied_array_indexes[0][j] #responsible X-grid.
            grid_y = occupied_array_indexes[1][j] #responsible Y-grid.
            anchor_index = occupied_array_indexes[2][j] #responsible anchor index.

            #get the prediction confidence.
            pred_confidence = predicted_arrays[grid_x][grid_y][anchor_index][0]
            #center coordinates of the predicted box.
            center_x = (predicted_arrays[grid_x][grid_y][anchor_index][1]*subsampled_ratio) + (grid_x*subsampled_ratio)
            center_y = (predicted_arrays[grid_x][grid_y][anchor_index][2]*subsampled_ratio) + (grid_y*subsampled_ratio)

            width = (anchors_list[grid_x][grid_y][anchor_index][3])*(math.e**(predicted_arrays[grid_x][grid_y][anchor_index][3]))
            height = (anchors_list[grid_x][grid_y][anchor_index][4])*(math.e**(predicted_arrays[grid_x][grid_y][anchor_index][4]))

            x1 = center_x - width/2
            y1 = center_y - height/2
            x2 = center_x + width/2
            y2 = center_y + height/2

            if ground_truth_mode:

                class_index = predicted_arrays[grid_x][grid_y][anchor_index][5]

            else:
                class_index = np.argmax(predicted_arrays[grid_x][grid_y][anchor_index][5:])

            transformed_values.append([pred_confidence, x1, y1, x2, y2, class_index])

            #get the responsible anchor and transform the values.
            res_anchor = anchors_list[grid_x][grid_y][anchor_index]
            anchor_x1 = res_anchor[1] - res_anchor[3]/2
            anchor_y1 = res_anchor[2] - res_anchor[4]/2
            anchor_x2 = res_anchor[1] + res_anchor[3]/2
            anchor_y2 = res_anchor[2] + res_anchor[4]/2
            anchor_values.append([anchor_x1, anchor_y1, anchor_x2, anchor_y2])

        entire_batch_transformed_values.append(transformed_values)
        entire_anchor.append(anchor_values)

    # return np.asarray(entire_anchor, dtype=np.float32)
    return np.asarray(entire_batch_transformed_values, dtype=np.float32)
