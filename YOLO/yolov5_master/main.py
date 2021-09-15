
from YOLO.yolov5_master.train import train_main,train_parse_opt
from YOLO.yolov5_master.detect import detect_main,detect_parse_opt

def train_():
    # 初始化参数列表
    t_opt = train_parse_opt()
    """
    重设自定义参数,进行模型训练
    Usage-命令行使用方式:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
    Usage-IDE使用方式：直接在下面对应位置进行修改
    """
    # 数据集配置文件
    # t_opt.data = 'data/voc_tower.yaml'
    t_opt.data = 'data/coco128.yaml'
    # 模型配置文件
    # t_opt.cfg = 'models/yolov5s_tower.yaml'
    t_opt.cfg = 'models/yolov5s.yaml'
    # 预训练权重
    # weights/yolov5l.pt,yolov5l6.pt,yolov5m.pt,yolov5m6.pt,yolov5s6.pt,yolov5x.pt,yolov5x6.pt
    t_opt.weight = 'weights/yolov5s.pt'
    # 设置单次训练所选取的样本数
    t_opt.batch_size = 16
    # 设置训练样本训练的迭代次数
    t_opt.epochs = 100
    # 设置线程数
    t_opt.workers = 4
    # 训练结果的文件名称
    t_opt.name = 'coco128_yolov5s'

    """开始训练"""
    train_main(t_opt)

def detect_():
    # 初始化参数列表
    d_opt = detect_parse_opt()
    """
    重设自定义参数,进行预测推理
    Usage-命令行使用方式:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
    Usage-IDE使用方式：直接在下面对应位置进行修改
    """
    # 图像/图像集合/视频的源路径,内部自动文件类型进行判断
    # d_opt.source = '/home/hxzh02/文档/航拍数据集/VOCdevkit_tower_part/JPEGImages/2040.jpg'
    d_opt.source = '/home/hxzh02/MyGithub/TrainNetHub/YOLO/yolov5_master/data/images/'
    # 设置进行预测推理使用的权重模型文件
    d_opt.weights = '/home/hxzh02/MyGithub/TrainNetHub/YOLO/yolov5_master/runs/train/coco128_yolov5s2/weights/best.pt'
    # d_opt.weights = '/home/hxzh02/PycharmProjects/TrainNetHub/YOLO/yolov5_master/runs/train/tower_yolov5s16/weights/best.pt'
    # 设置是否需要预览
    d_opt.view_img = False

    """开始预测推理"""
    detect_main(d_opt)


if __name__ == '__main__':
    # 模型训练
    train_()

    # 模型预测
    detect_()
