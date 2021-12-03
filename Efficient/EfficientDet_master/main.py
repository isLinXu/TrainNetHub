# -*- coding:utf-8  -*-
'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-11-7 上午11:56
@desc: 执行主函数
'''
from Efficient.EfficientDet_master.train import get_train_args, train
from Efficient.EfficientDet_master.detect import get_detect_args, predict

def train_():
    t_opt = get_train_args()
    # ---------------------------------------------------#
    #   训练参数配置
    # ---------------------------------------------------#
    # 设置项目名称
    project_name = 'birdview_vehicles'
    t_opt.project = project_name
    # 设置数据集路径
    t_opt.data_path = 'datasets/'
    # 设置置信度
    t_opt.compound_coef = 0
    # 设置批次大小
    t_opt.batch_size = 12
    # 设置轮次数量
    t_opt.num_epochs = 3
    # 设置训练部位 True/False
    t_opt.head_only = True
    # 设置学习率
    t_opt.lr = 5e-3
    # 设置预训练模型路径
    t_opt.load_weights = 'weights/efficientdet-d0.pth'
    # 设置保存checkpoint的step
    t_opt.save_interval = 100
    # 设置模型保存路径
    t_opt.saved_path = 'runs/'
    # 设置日志保存路径
    t_opt.log_path = 'logs/'

    # 开始进行训练
    train(t_opt)


def detect_():
    d_opt = get_detect_args()
    # ---------------------------------------------------#
    #   推理预测参数配置
    # ---------------------------------------------------#
    # 设置数据集路径
    d_opt.data_path = '/home/hxzh02/MyGithub/TrainNetHub/Efficient/EfficientDet_master/datasets/smoke_coco/val/'
    # 设置图片名称与格式
    d_opt.img_path = 'smog_test_409.jpg'
    d_opt.save_imgpath = 'runs/detect/'
    # 设置置信度
    d_opt.compound_coef = 0
    # 设置模型文件路径
    d_opt.weight_file_name = 'logs/smoke_coco/efficientdet-d0_9_200.pth'
    # 设置阈值
    d_opt.threshold = 0.2
    # 设置类别列表
    d_opt.obj_list = ['smoke']

    # 开始进行预测
    predict(d_opt)


if __name__ == '__main__':
    # 训练模型
    train_()
    # 推理预测
    # detect_()
