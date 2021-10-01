import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


import YOLO.yolov3_master.val as val # import val.py to get mAP after each epoch
from YOLO.yolov3_master.models.experimental import attempt_load
from YOLO.yolov3_master.models.yolo import Model
from YOLO.yolov3_master.utils.autoanchor import check_anchors
from YOLO.yolov3_master.utils.datasets import create_dataloader
from YOLO.yolov3_master.utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
     strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from YOLO.yolov5_master.utils.metrics import fitness
from YOLO.yolov3_master.utils.google_utils import attempt_download
from YOLO.yolov3_master.utils.loss import ComputeLoss
from YOLO.yolov3_master.utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from YOLO.yolov3_master.utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel
from YOLO.yolov3_master.utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

logger = logging.getLogger(__name__)

def train_parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov3.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    opt = parser.parse_args()
    return opt


def train(hyp, opt, device, tb_writer=None):
    '''
    训练主体
    :param hyp:
    :param opt:
    :param device:
    :param tb_writer:
    :return:
    '''
    global ckpt
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    '''
    创建目录，设置模型、txt等保存的路径
    '''
    # Directories
    # 获取记录训练日志的路径
    '''
    训练日志包括：权重、tensorboard文件、超参数hyp、设置的训练参数opt(也就是epochs,batch_size等),result.txt
    result.txt包括: 占GPU内存、训练集的GIOU loss, objectness loss, classification loss, 总loss, 
    targets的数量, 输入图片分辨率, 准确率TP/(TP+FP),召回率TP/P ; 
    测试集的mAP50, mAP@0.5:0.95, box loss, objectness loss, classification loss.
    还会保存batch<3的ground truth
    '''
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    # 设置保存权重的路径
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    '''
    配置:画图开关,cuda,种子,读取数据集相关的yaml文件
    '''
    # Configure
    # 是否绘制训练、测试图片、指标图等，使用进化算法则不绘制
    plots = not opt.evolve  # create plots
    # 判断当前训练设备环境
    cuda = device.type != 'cpu'
    # 设置固定随机种子
    init_seeds(2 + rank)

    # 加载数据配置信息
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # data dict

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

    # 根据类别数量制定不同策略
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check
    is_coco = opt.data.endswith('coco.yaml') and nc == 80  # COCO dataset

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']

    # Freeze
    '''
    冻结层
    '''
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    '''
    优化器设置
    '''
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # 优化器参数组
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    # 设置biases的优化方式
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    # 打印优化信息
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2


    '''
    设置学习率策略:两者可供选择，线性学习率和余弦退火学习率
    参考论文：https://arxiv.org/pdf/1812.01187.pdf
    参考文档：https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    '''
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    # 是否使用线性学习率衰减，默认还是使用的余弦退火衰减
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        # 设置学习率衰减，这里为余弦退火方式进行衰减
        # 就是根据one_cycle中定义的公式，lf,epoch和超参数hyp['lrf']进行衰减
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    '''
    设置ema（指数移动平均）:目的是为了收敛的曲线更加平滑
    '''
    # EMA
    # 为模型创建EMA指数滑动平均,如果GPU进程数大于1,则不创建
    ema = ModelEMA(model) if rank in [-1, 0] else None

    '''
    (中断后)重新开始
    继续接着训练,需要加载优化器,ema模型,训练结果txt,周期
    '''
    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        # 加载优化器与best_fitness
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']


        '''
        EMA&
        滑动平均exponential moving average，指数加权平均
        给W和b使用EMA，就是防止训练过程遇到异常数据或者随机跳跃影响训练效果的，
        让W和b维持相对稳定。
        '''
        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        # 获取计算结果
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        # 加载训练的轮次
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        """
        如果新设置epochs小于加载的epoch，
        则视新设置的epochs为需要再训练的轮次数而不再是总的轮次数
        """
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    '''
    模型默认的下采样倍率model.stride: [8,16,32]
    gs代表模型下采样的最大步长: 后续为了保证输入模型的图片宽高是最大步长的整数倍
    nl代表模型输出的尺度,默认为3个尺度, 分别下采样8倍，16倍，32倍.nl=3
    imgsz, imgsz_test代表训练和测试的图片大小，比如opt.img_size=[640,480]，那么训练图片的最大边为640,测试图片最大边为480
    如果opt.img_size=[640],那么自动补成[640,640]
    当然比如这边imgsz是640,那么训练的图片是640*640吗，不一定，具体看设置的方式，默认是padding成正方形进行训练的.
    '''
    # Image sizes
    # 获取模型总步长和模型输入图片分辨率
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # 获取模型FPN层数
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    # 检查输入图片分辨率确保能够整除总步长gs
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    '''
     多卡训练
     分布式训练,参照:https://github.com/ultralytics/yolov5/issues/475
     DataParallel模式,仅支持单机多卡
     rank为进程编号, 这里应该设置为rank=-1则使用DataParallel模式
     rank=-1且gpu数量=1时,不会进行分布式
     '''
    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # 使用跨卡同步BN
    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    '''
    加载数据集
    '''
    # Trainloader
    # 创建训练集对象加载器dataloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    # 最大类别数量
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    # 批次数量
    nb = len(dataloader)  # number of batches
    # 断言：mlc<nc，若不满足条件则直接返回错误
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    # 初始情况
    if rank in [-1, 0]:
        # 创建测试集dataloader
        testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]

        if not opt.resume:
            # 将所有样本的标签拼接到一起shape为(total, 5)，统计后做可视化
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # 获得所有样本的类别
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            # 根据上面的统计对所有样本的类别，中心点xy位置，长宽wh做可视化
            if plots:
                plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            """
            计算默认锚点anchor与数据集标签框的长宽比值
            标签的长h宽w与anchor的长h_a宽w_a的比值, 即h/h_a, w/w_a都要在(1/hyp['anchor_t'], hyp['anchor_t'])是可以接受的
            如果标签框满足上面条件的数量小于总数的98%，则根据k-mean算法聚类新的锚点anchor
            """
            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    # 如果rank不等于-1,则使用DistributedDataParallel模式
    # local_rank为gpu编号,rank为进程,
    # 例如rank=3，local_rank=0 表示第 3 个进程内的第 1 块 GPU。
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    '''
    模型参数的一些调整
    '''
    # Model parameters
    # 根据自己数据集的类别数和网络FPN层数设置各个损失的系数
    # 计算box loss，也即为xywh部分带来的误差,其中nl为网络FPN层数
    hyp['box'] *= 3. / nl  # scale to layers
    # 计算class loss,也即为类别带来的误差
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    # 计算obj loss,也即为置信度带来的误差
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers

    # 标签平滑
    hyp['label_smoothing'] = opt.label_smoothing
    # 设置类别数，超参数
    model.nc = nc  # attach number of classes to model
    # 给模型附加超参数
    model.hyp = hyp  # attach hyperparameters to model
    # iou loss ratio
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    # 根据labels初始化图片采样权重
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    # 获取类别的名字
    model.names = names

    '''
    开始训练    
    '''
    # Start training
    # 记录开始训练时间
    t0 = time.time()
    # 获取热身训练的迭代次数
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training

    # 初始化mAP和results
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

    """
    设置学习率衰减所进行到的轮次，
    目的是打断训练后，--resume接着训练也能正常的衔接之前的训练进行学习率衰减
    """
    scheduler.last_epoch = start_epoch - 1  # do not move
    # 通过torch自带的api设置混合精度训练
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class

    """
    打印训练和测试输入图片分辨率
    加载图片时调用的cpu进程数
    日志目录
    从哪个epoch开始训练
    """
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        # 更新图像权重(可选，仅限单gpu)
        if opt.image_weights:
            """
            如果设置进行图片采样策略，
            则根据前面初始化的图片采样权重model.class_weights以及maps配合每张图片包含的类别数
            通过random.choices生成图片索引indices从而进行采样
            """
            # Generate indices
            if rank in [-1, 0]:
                # class weights
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
                # image weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)
                # rand weighted idx
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # 更新边界
        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        # 初始化训练时打印的平均损失信息
        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            # DDP模式下打乱数据, ddp.sampler的随机采样数据是基于epoch+seed作为随机种子，
            # 每次epoch不同，随机种子就不同
            dataloader.sampler.set_epoch(epoch)

        '''
        Log打印相关更新参数
        '''
        # 设置打印Log列表名称
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))

        if rank in [-1, 0]:
            # tqdm 创建进度条，方便训练时 信息的展示
            pbar = tqdm(pbar, total=nb)  # progress bar

        # zero the parameter gradients
        # 梯度置零，也就是把loss关于weight的导数变成0.
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            # 计算迭代的次数iteration
            ni = i + nb * epoch  # number integrated batches (since train start)
            # 符号位数转换 uint8=>8位float32无符号整型转为32位双精度浮点
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            """
            预热训练(前nw次迭代)
            在前nw次迭代中，根据以下方式选取accumulate和学习率
            """
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    """
                    bias的学习率从0.1下降到基准学习率lr*lf(epoch)，
                    其他的参数学习率从0增加到lr*lf(epoch).
                    lf为上面设置的余弦退火的衰减函数
                    """
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            # 设置多尺度训练，从imgsz * 0.5, imgsz * 1.5 + gs随机选取尺寸
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            '''
            前向传播
            '''
            # Forward
            # 混合精度
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                # 计算损失，包括分类损失，objectness损失，框的回归损失
                # loss为总损失值，loss_items为一个元组，包含分类损失，objectness损失，框的回归损失和总损失
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if rank != -1:
                    # 平均不同gpu之间的梯度
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.
            '''
            反向传播
            '''
            # Backward
            scaler.scale(loss).backward()


            '''
            最优化
            模型反向传播累积次数之后再根据累积的梯度更新一次参数
            '''
            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Log 日志信息打印
            # 打印显存，进行的轮次，损失，target的数量和图片的size等信息
            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                # 进度条显示以上信息
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 3:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    if tb_writer:
                        tb_writer.add_graph(torch.jit.trace(de_parallel(model), imgs, strict=False), [])  # model graph
                        # tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # 进行学习率衰减
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            # 判断该epoch是否为最后一轮
            final_epoch = epoch + 1 == epochs
            # 对测试集进行测试，计算mAP等指标
            # 测试时使用的是EMA模型
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                results, maps, times = val.test(data_dict,
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 save_json=is_coco and final_epoch,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco)

            # Write
            '''
            将附加指标与loss写入result_file
            '''
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            # 更新best_fitness
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            """
            保存模型，这里将model与ema都进行保存，同时还保存了epoch，results，optimizer等信息，
            """
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}


                # Save last, best and delete
                # 保存last.pt,best.pt
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        logger.info(f'{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n')
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})

        if not opt.evolve:
            if is_coco:  # COCO dataset
                for m in [last, best] if best.exists() else [last]:  # speed, mAP tests
                    results, _, _ = test.test(opt.data,
                                              batch_size=batch_size * 2,
                                              imgsz=imgsz_test,
                                              conf_thres=0.001,
                                              iou_thres=0.7,
                                              model=attempt_load(m, device).half(),
                                              single_cls=opt.single_cls,
                                              dataloader=testloader,
                                              save_dir=save_dir,
                                              save_json=True,
                                              plots=False,
                                              is_coco=is_coco)

            # Strip optimizers
            """
            模型训练完后，strip_optimizer函数将除了模型model或者ema之外的所有东西去除；
            并且对模型进行model.half(), 将Float32的模型->Float16，
            可以减少模型大小，提高inference速度
            """
            for f in last, best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers
            if wandb_logger.wandb:  # Log the stripped model
                wandb_logger.wandb.log_artifact(str(best if best.exists() else last), type='model',
                                                name='run_' + wandb_logger.wandb_run.id + '_model',
                                                aliases=['latest', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()

    # 释放显存
    torch.cuda.empty_cache()
    return results

def train_main(t_opt):
    # 初始化参数列表

    # Set DDP variables
    """
    设置DDP模式的参数
    world_size:表示全局进程个数
    global_rank:进程编号
    """
    t_opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    t_opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(t_opt.global_rank)
    if t_opt.global_rank in [-1, 0]:
        check_git_status()
        check_requirements(exclude=('pycocotools', 'thop'))

    # Resume
    wandb_run = check_wandb_resume(t_opt)

    '''
    判断是否继续训练
    '''
    if t_opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = t_opt.resume if isinstance(t_opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = t_opt.global_rank, t_opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        t_opt.cfg, t_opt.weights, t_opt.resume, t_opt.batch_size, t_opt.global_rank, t_opt.local_rank = \
            '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # 获取超参数列表
        # t_opt.hyp = t_opt.hyp or ('hyp.finetune.yaml' if t_opt.weights else 'hyp.scratch.yaml')
        # 检查配置文件信息
        t_opt.data, t_opt.cfg, t_opt.hyp = check_file(t_opt.data), check_file(t_opt.cfg), check_file(t_opt.hyp)  # check files
        assert len(t_opt.cfg) or len(t_opt.weights), 'either --cfg or --weights must be specified'
        # 扩展image_size为[image_size, image_size]一个是训练size，一个是测试size
        t_opt.img_size.extend([t_opt.img_size[-1]] * (2 - len(t_opt.img_size)))  # extend to 2 sizes (train, test)
        t_opt.name = 'evolve' if t_opt.evolve else t_opt.name
        # 根据opt.logdir生成目录
        t_opt.save_dir = str(increment_path(Path(t_opt.project) / t_opt.name, exist_ok=t_opt.exist_ok | t_opt.evolve))

    # DDP mode
    # DDP mode  -->  支持多机多卡、分布式训练
    t_opt.total_batch_size = t_opt.batch_size
    # 选择设备
    device = select_device(t_opt.device, batch_size=t_opt.batch_size)
    if t_opt.local_rank != -1:
        assert torch.cuda.device_count() > t_opt.local_rank
        # 根据gpu编号选择设备
        torch.cuda.set_device(t_opt.local_rank)
        device = torch.device('cuda', t_opt.local_rank)
        # 初始化进程组
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert t_opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        assert not t_opt.image_weights, '--image-weights argument is not compatible with DDP training'
        # 将总批次按照进程数分配给各个gpu
        t_opt.batch_size = t_opt.total_batch_size // t_opt.world_size

    # Hyperparameters
    with open(t_opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps


    # Train
    '''
    开始训练
    这里先进行一个判断：
    若不进行超参数进化，则直接调用train()函数，开始训练
    否则，对超参数进行更新后再进行训练
    '''
    logger.info(t_opt)
    if not t_opt.evolve:
        tb_writer = None  # init loggers
        if t_opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {t_opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(t_opt.save_dir)  # Tensorboard
        # 调用train（）
        train(hyp, t_opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        # 超参数进化元数据
        '''
        'lr0':初始化学习率
        'lrf':周期性学习率
        'momentum':动量(使用SGD/Adam beta1)
                   动量的引入就是为了加快学习过程，特别是对于高曲率、小但一致的梯度，
                   或者噪声比较大的梯度能够很好的加快学习过程。
                   动量的主要思想是积累了之前梯度指数级衰减的移动平均（前面的指数加权平均）。
        'weight_decay':权重衰减优化器，神经网络经常加入weight decay来防止过拟合
        'warmup_epochs':预热周期
        'warmup_momentum'：预热初始化动量
        'warmup_bias_lr':预热偏置学习率
        'box'：预测框位置box的loss
        'cls'：类别误差loss
            如果是单类的情况，cls loss=0
            如果是多类的情况，也分两个模式：
            如果采用default模式，使用的是BCEWithLogitsLoss计算class loss。
            如果采用CE模式，使用的是CrossEntropy同时计算obj loss和cls loss。
        'cls_pw':二分类交叉熵（Binary Cross Entropy）损失函数正向权重
        'obj':obj代表置信度，即该bounding box中是否含有物体的概率。
              置信度带来的误差，也就是obj带来的loss(按像素缩放)
        'obj_pw':关于obj置信度的BCELoss损失函数反向权重
        'iou_t': #IoU训练阈值
                IoU 的全称为交并比（Intersection over Union）。
                顾名思义，IoU 计算的是 “预测的边框” 和 “真实的边框” 的交集和并集的比值。
        'anchor_t':anchor机制下的多锚定阈值
        'anchors':每个输出栅格的定位（忽略0）
        'fl_gamma':Focal loss gamma伽马参数(设置有效伽马参数默认为1.5)
                   Focal loss主要是为了解决one-stage目标检测中正负样本比例严重失衡的问题。
                   该损失函数降低了大量简单负样本在训练中所占的权重，也可理解为一种困难样本挖掘。
        'hsv_h':图像HSV-色调(Hue)色调增强（分数fraction）
        'hsv_s':图像HSV-饱和度(Saturation)增强（分数fraction）
        'hsv_v':图像HSV-明度(Value)增强（分数fraction）
        'degrees':图像旋转(+/- deg角度)
        'translate':图像位移 (+/- 分数fraction)
        'scale':图像放缩(+/- 增益gain)
        'shear':图像错切(+/- deg角度)
        'perspective':图像透视变换(+/- fraction), range 0-0.001
        'flipud':按照一定概率进行图像上下翻转
        'fliplr':按照一定概率进行图像左右翻转
        'mosaic':按照一定概率进行图像混合(概率)
        'mixup':按照一定概率进行图像混合(概率)
        'copy_paste':# 按照一定概率进行分割复制粘贴
        '''
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        assert t_opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        t_opt.notest, t_opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(t_opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if t_opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        """
        这里的进化算法是：根据之前训练时的hyp来确定一个base hyp再进行突变；
        如何根据？通过之前每次进化得到的results来确定之前每个hyp的权重
        有了每个hyp和每个hyp的权重之后有两种进化方式；
        1.根据每个hyp的权重随机选择一个之前的hyp作为base hyp，random.choices(range(n), weights=w)
        2.根据每个hyp的权重对之前所有的hyp进行融合获得一个base hyp，(x * w.reshape(n, 1)).sum(0) / w.sum()
        evolve.txt会记录每次进化之后的results+hyp
        每次进化时，hyp会根据之前的results进行从大到小的排序；
        再根据fitness函数计算之前每次进化得到的hyp的权重
        再确定哪一种进化方式，从而进行进化
        """
        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                # 选择进化方式
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                # 加载evolve.txt
                x = np.loadtxt('evolve.txt', ndmin=2)
                # 选取至多前5次进化的结果
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                # 根据results计算hyp的权重
                w = fitness(x) - fitness(x).min()  # weights
                # 根据不同进化方式获得base hyp
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                # 超参数进化
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                # 设置突变
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                # 将突变添加到base hyp上
                # [i+7]是因为x中前七个数字为results的指标(P, R, mAP, F1, val_losses=(box, obj, cls))，之后才是超参数hyp
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            '''
            修剪hyp在规定范围里
            为了防止突变过程，导致参数出现明显不合理的范围，
            需要用一个范围进行框定，将超出范围的内容剪切掉。
            具体方法如下
            '''
            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), t_opt, device)

            """
            写入results和对应的hyp到evolve.txt
            evolve.txt文件每一行为一次进化的结果
            一行中前七个数字为(P, R, mAP, F1, val_losses=(box, obj, cls))，之后为hyp
            保存hyp到yaml文件
            """
            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, t_opt.bucket)

        # Plot results
        # 可视化结果 plot_evolve（utils/plots.py）
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')

if __name__ == '__main__':
    # 初始化参数列表
    t_opt = train_parse_opt()
    # 重设自定义参数
    t_opt.data = '/home/hxzh02/MyGithub/TrainNetHub/YOLO/yolov3_master/data/coco128.yaml'
    t_opt.cfg = '/home/hxzh02/MyGithub/TrainNetHub/YOLO/yolov3_master/models/yolov3.yaml'
    t_opt.weights = '/home/hxzh02/MyGithub/TrainNetHub/YOLO/yolov3_master/weights/yolov3.pt'
    t_opt.batch_size = 8
    t_opt.epochs = 100
    t_opt.workers = 4
    t_opt.name = 'tower_yolov3t'
    train_main(t_opt)