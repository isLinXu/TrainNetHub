import cv2
import sys
import os


class PackageProjectUtil:
    @staticmethod
    def project_root_path(project_name=None, print_log=True):
        """
        获取当前项目根路径
        :param project_name: 项目名称
        :param print_log: 是否打印日志信息
        :return: 指定项目的根路径
        """
        p_name = 'TrainNetHub' if project_name is None else project_name
        project_path = os.path.abspath(os.path.dirname(__file__))
        # Windows
        if project_path.find('\\') != -1: separator = '\\'
        # Mac、Linux、Unix
        if project_path.find('/') != -1: separator = '/'

        root_path = project_path[:project_path.find(f'{p_name}{separator}') + len(f'{p_name}{separator}')]
        if print_log: print(f'当前项目名称：{p_name}\r\n当前项目根路径：{root_path}')
        return root_path


# 将当前项目目录添加至Python编译器路径(兼容python命令行运行方式)
sys.path.append(PackageProjectUtil.project_root_path())
# 当前目录
rpath = sys.path[0]

from YOLO.yolov5_master.train import train_main, train_parse_opt
from YOLO.yolov5_master.detect import detect_main, detect_parse_opt


def train_(object_name, models_name='yolov5s'):
    # 初始化参数列表
    t_opt = train_parse_opt()
    """
    重设自定义参数,进行模型训练
    Usage-命令行使用方式:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
    Usage-IDE使用方式：直接在下面对应位置进行修改
    """
    # 数据集配置文件
    t_opt.data = rpath + '/data/' + 'custom/' + 'custom_' + object_name + '.yaml'
    # 模型配置文件
    t_opt.cfg = rpath + '/models/custom/' + models_name + '_' + object_name + '.yaml'
    # 预训练权重
    # weights/yolov5l.pt,yolov5l6.pt,yolov5m.pt,yolov5m6.pt,yolov5s6.pt,yolov5x.pt,yolov5x6.pt
    t_opt.weights = rpath + '/weights/' + models_name + '.pt'
    # 设置单次训练所选取的样本数
    t_opt.batch_size = 8
    # 设置训练样本训练的迭代次数
    t_opt.epochs = 300
    # 设置线程数
    t_opt.workers = 4
    # 训练结果的文件名称
    t_opt.name = models_name + '_' + object_name

    """开始训练"""
    train_main(t_opt)


def detect_(object_name, models_name='yolov5s'):
    # 初始化参数列表
    d_opt = detect_parse_opt()
    """
    重设自定义参数,进行预测推理
    Usage-命令行使用方式:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
    Usage-IDE使用方式：直接在下面对应位置进行修改
    """
    # 图像/图像集合/视频的源路径,内部自动文件类型进行判断
    # d_opt.source = rpath +'/data/images/bus.jpg'
    # d_opt.source = '/media/hxzh02/SB@home/hxzh/Dataset/11-5电塔照片视频/照片/'
    # d_opt.source = '/media/hxzh02/SB@home/hxzh/Dataset/无人机相关数据集合集/3-输电线路异物数据集（VOC）/foreignbody_dataset_part1/images/val/'
    # d_opt.source = '/home/hxzh02/MyGithub/TrainNetHub/YOLO/yolov5_master/data/VOCdevkit_tower_part/VOC2007/JPEGImages'
    # d_opt.source = '/home/hxzh02/MyGithub/TrainNetHub/YOLO/yolov5_master/data/datasets_smoke/VOC2007/JPEGImages'
    # d_opt.source = '/media/hxzh02/SB@home/hxzh/Dataset/无人机相关数据集合集/7-输电线路绝缘子数据集VOC/dataset_insulator/VOC2007/JPEGImages/'
    # d_opt.source = '/media/hxzh02/SB@home/hxzh/Dataset/无人机相关数据集合集/5-安全帽数据集5000张/dataset_safetyHat/images/val/'

    d_opt.source = '/media/hxzh02/SB@home/hxzh/Dataset/Plane_detect_datasets/VOCdevkit_towerbody_detect/images/val'
    # d_opt.source = '/media/hxzh02/SB@home/hxzh/Dataset/输电杆塔照片素材'
    # d_opt.source = '/home/hxzh02/文档/test_image/towerupdown'
    # 设置进行预测推理使用的权重模型文件
    # d_opt.weights = rpath + '/runs/train/' + models_name + '_' + object_name + '/weights/best.pt'
    # d_opt.weights = '/home/hxzh02/MyGithub/TrainNetHub/YOLO/yolov5_master/runs/train/yolov5s_tower4/weights/best.pt'
    # /media/hxzh02/SB@home/hxzh/MyGithub/TrainNetHub/YOLO/yolov5_master/runs/train/yolov5s_tower_body2/weights/best.pt
    # d_opt.weights = '/media/hxzh02/SB@home/hxzh/MyGithub/TrainNetHub/YOLO/yolov5_master/runs/train/yolov5s_tower_only/weights/best.pt'
    d_opt.weights = '/media/hxzh02/SB@home/hxzh/MyGithub/TrainNetHub/YOLO/yolov5_master/runs/train/yolov5s_tower_body2/weights/best.pt'
    # 设置是否需要预览
    d_opt.view_img = False
    # 置信度设置
    d_opt.conf_thres = 0.5
    # 边界框线条粗细
    d_opt.line_thickness = 4
    cv2.waitKey()
    """开始预测推理"""
    detect_main(d_opt)


if __name__ == '__main__':
    # 设置训练任务/生成模型名称
    object_list = [
        'tower','foreignbody',
        'smoke','insulator',
        'helmet','firesmoke',
        'plane_all','tower_only',
        'tower_head'
    ]

    object_name = 'tower_body'
    # 模型选择
    models_list = [
        'yolov5n', 'yolov5n6',
        'yolov5s', 'yolov5s6',
        'yolov5m', 'yolov5m6',
        'yolov5l', 'yolov5m6',
        'yolov5x', 'yolov5x6',
    ]
    models_name = models_list[2]
    # 模型训练
    train_(object_name=object_name,models_name=models_name)

    # 模型预测
    # detect_(object_name)