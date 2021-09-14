
from YOLO.yolov5_master.train import main,parse_opt

if __name__ == '__main__':
    opt = parse_opt(known=False)
    main(opt)