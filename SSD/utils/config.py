# -*- coding:utf-8 -*-
# power by Mr.Li
# 设置默认参数
import os.path


class DefaultConfig():
    env = 'SSD_'  # visdom 环境的名字
    visdom = True  # 是否可视化
    # 目前支持的网络
    model = 'vgg16'
    # VOC数据集根目录,该文件夹下有两个子文件夹.一个叫VOC2007,一个叫VOC2012
    voc_data_root = '/media/hxzh02/SB@home/hxzh/Dataset/VOCdevkit/'

    # 基础网络，即特征提取网络（去掉全连接的预训练模型vgg16）
    basenet = '/home/bobo/windowsPycharmProject/SSD_pytorch/checkpoint/vgg16_reducedfc.pth'  # 应为全路径 预训练好的去掉全连接层的vgg16模型
    batch_size = 8  # 训练集的batch size
    start_iter = 0  # 训练从第几个item开始
    num_workers = 4  # 加载数据时的线程数
    use_gpu = True  # user GPU or not
    lr = 0.001  # 初始的学习率
    momentum = 0.9  # 优化器的动量值
    weight_decay = 5e-4  # 随机梯度下降SGD的权重衰减
    gamma = 0.1  # Gamma update for SGD  学习率调整参数

    checkpoint_root = '/home/hxzh02/PycharmProjects/PytorchNetHub/SSD_pytorch/checkpoint'  # 保存模型的目录
    # load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    load_model_path = '/home/hxzh02/PycharmProjects/PytorchNetHub/SSD_pytorch/checkpoint/ssd300_VOC_100000.pth'
    # gets home dir cross platform
    HOME = os.path.expanduser("~")
    # 使边界框漂亮
    COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
              (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))
    MEANS = (104, 117, 123)
    # SSD300 配置
    voc = {
        'num_classes': 21,  # 分类类别20+背景1
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,  # 迭代次数
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'min_dim': 300,  # 当前SSD300只支持大小300×300的数据集训练
        'steps': [8, 16, 32, 64, 100, 300],  # 感受野，相对于原图缩小的倍数
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],  # 方差
        'clip': True,
        'name': 'VOC',
    }

    # 验证
    confidence_threshold = 0.01  # 检测置信度阈值  or 0.05
    top_k = 5  # 进一步限制要解析的预测数量
    cleanup = True  # 清除并删除eval后的结果文件
    # 保存验证的临时文件
    temp = '/home/hxzh02/PycharmProjects/PytorchNetHub/SSD_pytorch/temp/'
    annopath = os.path.join(voc_data_root, 'VOC2007', 'Annotations', '%s.xml')
    imgpath = os.path.join(voc_data_root, 'VOC2007', 'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(voc_data_root, 'VOC2007', 'ImageSets',
                              'Main', '{:s}.txt')

    # 测试
    temp_test = '/home/hxzh02/PycharmProjects/PytorchNetHub/SSD_pytorch/temp/'  # 保存测试集（VOC2007测试集）的网络预测结果

    # 预测，可视化一张预测图片
    test_img = '/home/hxzh02/PycharmProjects/PytorchNetHub/SSD_pytorch/temp/test.png'


# 初始化该类的一个对象
opt = DefaultConfig()
