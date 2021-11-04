'''
Loads the required dataset for both ImageNet model training and YOLOv2 model training.
'''
import dbm
import numpy as np
import torch
from torch.utils.data import Dataset
import cfg
from utils import cluster_bounding_boxes, generate_anchors, generate_training_data, imgnet_generate_data, imgnet_read_data, generate_test_data



class ToTensor:
    '''
    Transforms the images and labels from numpy array to tensors.
    '''
    def __init__(self, mode='train'):
        self.mode = mode

    def __call__(self, sample):

        if self.mode == 'train':
            image = sample['image']
            label = sample['label']

            image = image.transpose((2, 0, 1)) #pytorch requires the channel to be in the 1st dimension of the tensor,


            return {'image':torch.from_numpy(image.astype('float32')),
                    'label': torch.from_numpy(label)}
        else:

            image = sample['image']

            image = image.transpose((2, 0, 1))

            return {'image': torch.from_numpy(image.astype('float32'))}



class LoadDataset(Dataset):
    '''
    Wraps the loading of the dataset in PyTorch's DataLoader module.
    '''

    def __init__(self, resized_image_size, k=cfg.K, classes=cfg.CLASSES.copy(), list_images=cfg.LIST_IMAGES.copy(), list_annotations=cfg.LIST_ANNOTATIONS.copy(),
                 total_images=cfg.TOTAL_IMAGES, subsampled_ratio=cfg.SUBSAMPLED_RATIO, detection_conv_size=cfg.DETECTION_CONV_SIZE,
                 excluded_classes=cfg.EXCLUDED_CLASSES.copy(), anchor_box_write=cfg.ANCHOR_BOXES_STORE, transform=None):

        '''
        Initialize parameters and anchors using KMeans.
        '''

        self.resized_image_size = resized_image_size
        self.classes = classes
        self.list_images = list_images
        self.list_annotations = list_annotations
        self.total_images = total_images
        self.k = k
        self.subsampled_ratio = subsampled_ratio
        self.detection_conv_size = detection_conv_size
        self.excluded_classes = excluded_classes
        self.transform = transform
        self.anchor_boxes_write = anchor_box_write

        #get the top-k anchor sizes using modifed K-Means clustering.
        self.anchor_sizes = cluster_bounding_boxes(k=self.k, total_images=self.total_images, resized_image_size=self.resized_image_size,
                                                   list_annotations=cfg.LIST_ANNOTATIONS, classes=cfg.CLASSES, excluded_classes=cfg.EXCLUDED_CLASSES)

        #python dbm to store the anchor sizes for a specific training set for every image size.
        #the anchor sizes are necessary for the use of evaluation later as we would not have training data to perform clustering.
        yolo_db = dbm.open(cfg.YOLO_DB, 'c')
        #for every image size, different anchor set. Each set will be stored in database for the use of evaluation later.
        yolo_db[str(resized_image_size)] = str(self.anchor_sizes.tolist())
        yolo_db.close()

        self.anchors_list = generate_anchors(anchor_sizes=self.anchor_sizes,
                                             subsampled_ratio=self.subsampled_ratio, resized_image_size=self.resized_image_size)



    def __len__(self):
        '''
        Abstract method. Returns the total number of data.
        '''
        return self.total_images


    def __getitem__(self, idx):
        '''
        Abstract method. Returns the image and label for a single input with the index of `idx`.
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()


        image, label_array = generate_training_data(anchors_list=self.anchors_list,
                                        xml_file_path=self.list_annotations[idx], classes=self.classes, resized_image_size=self.resized_image_size,
                                    subsampled_ratio=self.subsampled_ratio, excluded_classes=self.excluded_classes, image_path=self.list_images[idx])

        sample = {'image':image,
                 'label':label_array}

        if self.transform:
            sample = self.transform(sample)

        return sample



class LoadTestData(Dataset):
    '''
    Loads the evaluation images.
    '''
    def __init__(self, resized_image_size, test_images_list=cfg.TEST_IMAGE_LIST.copy(), anchor_box_file=cfg.ANCHOR_BOXES_STORE,
                 subsampled_ratio=cfg.SUBSAMPLED_RATIO, transform=None):

        self.anchor_box_file = anchor_box_file
        self.subsampled_ratio = subsampled_ratio
        self.resized_image_size = resized_image_size
        self.test_images_list = test_images_list
        self.transform = transform

        anchor_file = open(self.anchor_box_file, 'r')
        anchor_tuples = anchor_file.read().replace('\n', '')

        assert(len(anchor_tuples) > 0, "Anchor data is empty!")

        self.all_anchor_sizes = dict(eval(anchor_tuples))
        self.anchor_sizes = np.asarray(eval(self.all_anchor_sizes[str(self.resized_image_size)]), dtype=np.float32)

        self.anchors_list = generate_anchors(anchor_sizes=self.anchor_sizes,
                                             subsampled_ratio=self.subsampled_ratio, resized_image_size=self.resized_image_size)


    def __len__(self):
        '''
        Abstract method. Returns the total number of data.
        '''

        return len(self.test_images_list)


    def __getitem__(self, idx):
        '''
        Abstract method. Returns the image and label for a single input with the index of `idx`.
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()


        image = generate_test_data(resized_image_size=self.resized_image_size,
                                                image_path=self.test_images_list[idx])

        sample = {'image':image}

        if self.transform:
            sample = self.transform(sample)

        return sample



class ImgnetLoadDataset(Dataset):
    '''
     Wraps the loading of the ImageNet dataset in PyTorch's DataLoader module.
    '''

    def __init__(self, resized_image_size, class_list, dataset_folder_path, transform=None):
        '''
        Initialize parameters and generate the images path list and corresponding labels.
        '''

        self.resized_image_size = resized_image_size
        self.class_list = class_list
        self.dataset_folder_path = dataset_folder_path
        self.transform = transform

        self.images_path_list, self.labels_list = imgnet_generate_data(folder_path=self.dataset_folder_path,
                                                                      class_list=self.class_list)


    def __len__(self):
        '''
        Returns the number of data in the list.
        '''

        return len(self.images_path_list)

    def __getitem__(self, idx):
        '''
        Read one single data.
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_array, label_array = imgnet_read_data(image_path=self.images_path_list[idx], class_idx=self.labels_list[idx],
                                                    resized_image_size=self.resized_image_size)

        sample = {
            'image':image_array,
            'label':label_array
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
