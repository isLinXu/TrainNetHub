# -*- coding:utf-8  -*-
'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-11-7 上午10:56
@desc: 推理检测可视化
'''
import os

import torch
from torch.backends import cudnn

import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from Efficient.EfficientDet_master.backbone import EfficientDetBackbone
from Efficient.EfficientDet_master.efficientdet.utils import BBoxTransform, ClipBoxes
from Efficient.EfficientDet_master.utils.utils import preprocess, invert_affine, postprocess

def get_detect_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-d', '--data_path', type=str, default='datasets/birdview_vehicles/val/')
    parser.add_argument('-i', '--img_path', type=str, default='1135.jpg', help='img_path')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-w', '--weight_file_name', type=str,
                        default='logs/birdview_vehicles/efficientdet-d0_9_2770.pth')
    parser.add_argument('-f', '--force_input_size', type=list, default=None)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--iou_threshold', type=float, default=0.2)
    parser.add_argument('--obj_list', type=list, default=['large-vehicle', 'small-vehicle'])
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--use_float16', type=bool, default=True)
    parser.add_argument('--fastest', type=bool, default=True)
    parser.add_argument('--benchmark', type=bool, default=True)
    parser.add_argument('--save_imgpath', type=str, default='runs/detect/')

    args = parser.parse_args()
    return args


def predict(opt):
    # 参数解析
    compound_coef = opt.compound_coef
    force_input_size = opt.force_input_size
    img_path = opt.data_path + opt.img_path
    weight_file_name = opt.weight_file_name
    threshold = opt.threshold
    iou_threshold = opt.iou_threshold
    obj_list = opt.obj_list

    use_cuda = opt.use_cuda
    use_float16 = opt.use_float16
    cudnn.fastest = opt.fastest
    cudnn.benchmark = opt.benchmark

    os.makedirs(opt.save_imgpath, exist_ok=True)

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),

                                 # replace this part with your project's anchor config
                                 ratios=[(0.7, 1.4), (1.0, 1.0), (1.5, 0.7)],
                                 scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    model.load_state_dict(torch.load(weight_file_name))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    out = invert_affine(framed_metas, out)

    for i in range(len(ori_imgs)):
        if len(out[i]['rois']) == 0:
            continue
        ori_imgs[i] = ori_imgs[i].copy()
        for j in range(len(out[i]['rois'])):
            (x1, y1, x2, y2) = out[i]['rois'][j].astype(np.int)
            cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[out[i]['class_ids'][j]]
            print('print',obj)
            score = float(out[i]['scores'][j])
            print('score', score)

            cv2.putText(ori_imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)

            cv2.imshow(str(i), ori_imgs[i])
            cv2.imwrite(opt.save_imgpath + opt.img_path, ori_imgs[i])
    cv2.waitKey(0)


if __name__ == '__main__':
    d_opt = get_detect_args()
    predict(d_opt)
