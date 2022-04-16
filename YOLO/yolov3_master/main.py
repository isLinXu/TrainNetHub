import cv2

from YOLO.yolov3_master.train import train_main,train_parse_opt
from YOLO.yolov3_master.detect import detect_main,detect_parse_opt

def train_():
    # 初始化参数列表
    t_opt = train_parse_opt()
    """
    重设自定义参数,进行模型训练
    Usage-命令行使用方式:
    $ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/NUsoVlDFqZg'  # YouTube video
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
    Usage-IDE使用方式：直接在下面对应位置进行修改
    """
    # 数据集配置文件
    t_opt.data = 'data/coco128.yaml'
    # 模型配置文件
    t_opt.cfg = 'models/coco128.yaml'
    # 预训练权重
    # weights/yolov3.pt, yolov3-spp.pt,yolov3-tiny.pt,yolov5l.pt
    t_opt.weights = 'weights/yolov3.pt'
    # 设置单次训练所选取的样本数
    t_opt.batch_size = 8
    # 设置训练样本训练的迭代次数
    t_opt.epochs = 300
    # 设置线程数
    t_opt.workers = 4
    # 训练结果的文件名称
    t_opt.name = 'coco128_yolov3'

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
    d_opt.source = '/home/hxzh02/MyGithub/TrainNetHub/YOLO/yolov3_master/data/images/zidane.jpg'
    # 设置进行预测推理使用的权重模型文件
    d_opt.weights = '/home/hxzh02/MyGithub/TrainNetHub/YOLO/yolov3_master/weights/yolov3.pt'
    # 设置是否需要预览
    d_opt.view_img = False
    # 置信度设置
    d_opt.conf_thres = 0.5
    # 边界框线条粗细
    d_opt.line_thickness = 1
    cv2.waitKey()
    """开始预测推理"""
    detect_main(d_opt)

if __name__ == '__main__':
    # 模型训练
    train_()

    # 模型预测
    # detect_()
