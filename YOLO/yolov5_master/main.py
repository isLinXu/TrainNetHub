
from YOLO.yolov5_master.train import train_main,train_parse_opt
from YOLO.yolov5_master.detect import detect_main,detect_parse_opt

def train_():
    t_opt = train_parse_opt()
    # 重设自定义参数
    t_opt.data = 'data/voc_tower.yaml'
    t_opt.cfg = 'models/yolov5s_tower.yaml'
    t_opt.weight = 'weights/yolov5s.pt'
    t_opt.batch_size = 16
    t_opt.epochs = 100
    t_opt.workers = 4
    t_opt.name = 'tower_yolov5s'
    train_main(t_opt)

def detect_():
    d_opt = detect_parse_opt()
    d_opt.source = '/home/hxzh02/文档/航拍数据集/VOCdevkit_tower_part/JPEGImages/2040.jpg'
    d_opt.weights = '/home/hxzh02/PycharmProjects/TrainNetHub/YOLO/yolov5_master/runs/train/tower_yolov5s16/weights/best.pt'
    d_opt.view_img = False
    detect_main(d_opt)


if __name__ == '__main__':
    # 模型训练
    train_()

    # 模型预测
    # detect_()
