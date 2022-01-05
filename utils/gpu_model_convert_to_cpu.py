import os
import numpy as np
import torch
import torch.nn as nn
# from model_collect import AlexNet_bn,LeNet_bn_224,ResNet,BasicBlock
from Net.ResNet import ResNet, BasicBlock, Bottleneck
from collections import OrderedDict

class_num = 21

# AlexNet_bn_net = AlexNet_bn(classes=class_num)
# LeNet_bn_net = LeNet_bn_224(classes=class_num)
ResNet_net = ResNet(Bottleneck, [3, 4, 23, 3], class_num)


# AlexNet_bn_path_model = "/home/cjh/work_file/code_hx/model_all/torch_models/model_manufacturer/pth_collector/rotary_and_light/AlexNet/model_state_dict_19.pkl"
# LeNet_bn_path_model = "/home/cjh/work_file/code_hx/model_all/torch_models/model_manufacturer/pth_collector/rotary_and_light/LetNet_bn/model_state_dict_15.pkl"
ResNet_path_model = "/home/cjh/work_file/code_hx/model_all/torch_models/model_manufacturer/pth_collector/rotary_light_press/ResNet101/model_state_dict_2.pkl"


# AlexNet_bn_dict_new_path = "/home/cjh/work_file/code_hx/model_all/torch_models/model_manufacturer/package_dir/AlexNet_dict.pkl"
# LeNet_bn_dict_new_path = "/home/cjh/work_file/code_hx/model_all/torch_models/model_manufacturer/package_dir/LeNet_dict.pkl"
ResNet_dict_new_path = "/home/cjh/work_file/code_hx/model_all/torch_models/model_manufacturer/package_dir/ResNet18_dict.pkl"



# AlexNet_state_dict_load = torch.load(AlexNet_bn_path_model, map_location="cpu")
# LeNet_state_dict_load = torch.load(LeNet_bn_path_model, map_location="cpu")
ResNet_state_dict_load = torch.load(ResNet_path_model, map_location="cpu")



# net.load_state_dict(state_dict_load)

# remove module.


new_state_dict = OrderedDict()
for k, v in ResNet_state_dict_load.items():
    namekey = k[7:] if k.startswith('module.') else k
    new_state_dict[namekey] = v
print("new_state_dict:\n{}".format(new_state_dict))


torch.save(new_state_dict, ResNet_dict_new_path)
# net.load_state_dict(new_state_dict)