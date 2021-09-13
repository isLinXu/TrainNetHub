"""Train a YOLOv5 model on a custom dataset

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import logging
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import math
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from .val import run # for end-of-epoch mAP
from .models.experimental import attempt_load
from .models.yolo import Model
from .utils.autoanchor import check_anchors
from .utils.datasets import create_dataloader
from .utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr, methods
from .utils.downloads import attempt_download
from .utils.loss import ComputeLoss
from .utils.plots import plot_labels, plot_evolve
from .utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel
from .utils.loggers.wandb.wandb_utils import check_wandb_resume
from .utils.metrics import fitness
from .utils.loggers import Loggers
from .utils.callbacks import Callbacks

# 获取日志的一个实例，其中__name__（当前模块的派生名称-->train）为日志记录的用例名
LOGGER = logging.getLogger(__name__)
# 查找名为LOCAL_RANK，RANK，WORLD_SIZE的环境变量，若存在则返回环境变量的值，若不存在则返回第二个参数（-1，默认None）
# https://pytorch.org/docs/stable/elastic/run.html 该网址有详细介绍
# rank和local_rank的区别： 两者的区别在于前者用于进程间通讯，后者用于本地设备分配,
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

'''
训练主体
'''
def train(hyp,  # path/to/hyp.yaml or hyp dictionary
          opt,
          device,
          callbacks=Callbacks()  # 回调机制
          ):
    global ckpt
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

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

    # 设置保存权重的路径
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    '''
    读取hyp(超参数)配置文件
    '''
    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    # 显示超参数
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    '''
    将本次运行的超参数(hyp),和选项操作(opt)给保存成yaml格式
    '''
    # Save run settings 保存hyp和opt
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    data_dict = None

    '''
    加载相关日志功能:如tensorboard,logger,wandb
    '''
    # Loggers
    # 设置wandb和tb两种日志, wandb和tensorboard都是模型信息，指标可视化工具
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        # W&B 初始化
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    '''
    配置:画图开关,cuda,种子,读取数据集相关的yaml文件
    '''
    # Config
    # 是否绘制训练、测试图片、指标图等，使用进化算法则不绘制
    plots = not evolve  # create plots
    # 判断当前训练设备环境
    cuda = device.type != 'cpu'
    # 设置固定随机种子
    init_seeds(1 + RANK)

    # 加载数据配置信息
    with torch_distributed_zero_first(RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    is_coco = data.endswith('coco.yaml') and nc == 80  # COCO dataset

    '''
    加载模型
    '''
    # Model
    pretrained = weights.endswith('.pt')
    # 采用预训练
    if pretrained:
        # 加载模型，从google云盘或github上自动下载模型
        # 但通常会下载失败，建议提前下载下来放进weights目录
        with torch_distributed_zero_first(RANK):
            weights = attempt_download(weights)  # download if not found locally
        # 加载检查点
        ckpt = torch.load(weights, map_location=device)  # load checkpoint

        """
        这里模型创建，可通过opt.cfg，也可通过ckpt['model'].yaml
        这里的区别在于是否是resume，resume时会将opt.cfg设为空，
        则按照ckpt['model'].yaml创建模型；
        这也影响着下面是否除去anchor的key(也就是不加载anchor)，
        如果resume，则加载权重中保存的anchor来继续训练；
        主要是预训练权重里面保存了默认coco数据集对应的anchor，
        如果用户自定义了anchor，再加载预训练权重进行训练，会覆盖掉用户自定义的anchor；
        所以这里主要是设定一个，如果加载预训练权重进行训练的话，就去除掉权重中的anchor，采用用户自定义的；
        如果是resume的话，就是不去除anchor，就权重和anchor一起加载， 接着训练；
        参考https://github.com/ultralytics/yolov5/issues/459
        所以下面设置了intersect_dicts，该函数就是忽略掉exclude中的键对应的值
        """

        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        # 如果opt.cfg存在(表示采用预训练权重进行训练)就设置去除anchor
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        # 显示加载预训练权重的的键值对和创建模型的键值对
        # 如果设置了resume，则会少加载两个键值对(anchors,anchor_grid)
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        # 创建模型，ch为输入图片通道
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    # Freeze
    """
    冻结模型层,设置冻结层名字即可。使得这些层在反向传播的时候不再更新权重,需要冻结的层,可以写在freeze列表中
    具体可以查看https://github.com/ultralytics/yolov5/issues/679
    但作者不鼓励冻结层,因为他的实验当中显示冻结层不能获得更好的性能,参照:https://github.com/ultralytics/yolov5/pull/707
    并且作者为了使得优化参数分组可以正常进行,在下面将所有参数的requires_grad设为了True
    其实这里只是给一个freeze的示例
    """
    freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False

    '''
    nbs为模拟的batch_size，也就是名义批次,比如实际批次为16,那么64/16=4,每4次迭代，才进行一次反向传播更新权重，可以节约显存。
    就比如默认的话上面设置的opt.batch_size为16,这个nbs就为64，
    也就是模型梯度累积了64/16=4(accumulate)次之后
    再更新一次模型，变相的扩大了batch_size
    '''
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # 根据accumulate设置权重衰减系数
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    '''
    设置优化器，权重weight使用了正则化,偏置bias则不使用正则化
    '''
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    # 将模型分成三组(weight、bn, bias, 其他所有参数)优化
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    # 选用优化器，并设置g0组的优化方式
    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    # 设置weight、bn的优化方式
    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    # 设置biases的优化方式
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    # 打印优化信息
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    del g0, g1, g2

    '''
    设置学习率策略:两者可供选择，线性学习率和余弦退火学习率
    参考论文：https://arxiv.org/pdf/1812.01187.pdf
    参考文档：https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    '''
    # Scheduler
    # 是否使用线性学习率衰减，默认还是使用的余弦退火衰减
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        # 设置学习率衰减，这里为余弦退火方式进行衰减
        # 就是根据one_cycle中定义的公式，lf,epoch和超参数hyp['lrf']进行衰减
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    '''
    设置ema（指数移动平均）:目的是为了收敛的曲线更加平滑
    '''
    # EMA
    # 为模型创建EMA指数滑动平均,如果GPU进程数大于1,则不创建
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    '''
    继续接着训练,需要加载优化器,ema模型,训练结果txt,周期
    '''
    # Resume
    # 初始化开始训练的epoch和最好的结果
    # best_fitness是以[0.0, 0.0, 0.1, 0.9]为系数并乘以[精确度, 召回率, mAP@0.5, mAP@0.5:0.95]再求和所得
    # 根据best_fitness来保存best.pt
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        # 加载优化器与best_fitness
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        # 加载训练的轮次
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Epochs
        # 加载训练的轮次
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'

        """
        如果新设置epochs小于加载的epoch，
        则视新设置的epochs为需要再训练的轮次数而不再是总的轮次数
        """
        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, csd

    '''
    模型默认的下采样倍率model.stride: [8,16,32]
    gs代表模型下采样的最大步长: 后续为了保证输入模型的图片宽高是最大步长的整数倍
    nl代表模型输出的尺度,默认为3个尺度, 分别下采样8倍，16倍，32倍.   nl=3
    imgsz, imgsz_test代表训练和测试的图片大小，比如opt.img_size=[640,480]，那么训练图片的最大边为640,测试图片最大边为480
    如果opt.img_size=[640],那么自动补成[640,640]
    当然比如这边imgsz是640,那么训练的图片是640*640吗，不一定，具体看你怎么设置，默认是padding成正方形进行训练的.
    '''
    # Image sizes
    # 获取模型总步长和模型输入图片分辨率
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # 获取模型FPN层数
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    # 检查输入图片分辨率确保能够整除总步长gs
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    '''
    多卡训练
    分布式训练,参照:https://github.com/ultralytics/yolov5/issues/475
    DataParallel模式,仅支持单机多卡
    rank为进程编号, 这里应该设置为rank=-1则使用DataParallel模式
    rank=-1且gpu数量=1时,不会进行分布式
    '''
    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # 使用跨卡同步BN
    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    '''
    加载数据集
    '''
    # Trainloader
    # 创建训练集对象加载器dataloader
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                              hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect, rank=RANK,
                                              workers=workers, image_weights=opt.image_weights, quad=opt.quad,
                                              prefix=colorstr('train: '))

    '''
    检验加载的数据集是否正确:  利用数据集标签中的最大类别<nc（类别数）
    如果大于类别数则表示有问题
    '''
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in [-1, 0]:
        # 创建测试集dataloader
        val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                       hyp=hyp, cache=None if noval else opt.cache, rect=True, rank=-1,
                                       workers=workers, pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:
            # 将所有样本的标签拼接到一起shape为(total, 5)，统计后做可视化
            labels = np.concatenate(dataset.labels, 0)
            # 获得所有样本的类别
            # c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            # 根据上面的统计对所有样本的类别，中心点xy位置，长宽wh做可视化
            if plots:
                plot_labels(labels, names, save_dir)
            """
            计算默认锚点anchor与数据集标签框的长宽比值
            标签的长h宽w与anchor的长h_a宽w_a的比值, 即h/h_a, w/w_a都要在(1/hyp['anchor_t'], hyp['anchor_t'])是可以接受的
            如果标签框满足上面条件的数量小于总数的98%，则根据k-mean算法聚类新的锚点anchor
            """
            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

        # 在每个训练前例行程序结束时触发所有已注册的回调
        callbacks.on_pretrain_routine_end()

    # DDP mode
    # 如果rank不等于-1,则使用DistributedDataParallel模式
    # local_rank为gpu编号,rank为进程,例如rank=3，local_rank=0 表示第 3 个进程内的第 1 块 GPU。
    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    '''
    模型参数的一些调整
    '''
    # Model parameters
    # 根据自己数据集的类别数和网络FPN层数设置各个损失的系数
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    # 标签平滑
    hyp['label_smoothing'] = opt.label_smoothing
    # 设置类别数，超参数
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    # 根据labels初始化图片采样权重
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    # 获取类别的名字
    model.names = names

    '''
    开始训练    
    '''
    # Start training
    t0 = time.time()
    # 获取热身训练的迭代次数
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    # 现在梯度累积不是ni % accumulate了，而是ni - last_opt_step >= accumulate, 本质上区别不大
    last_opt_step = -1
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
    # 初始化声明计算损失的实例
    compute_loss = ComputeLoss(model)  # init loss class

    """
    打印训练和测试输入图片分辨率
    加载图片时调用的cpu进程数
    日志目录
    从哪个epoch开始训练
    """
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            """
            如果设置进行图片采样策略，
            则根据前面初始化的图片采样权重model.class_weights以及maps配合每张图片包含的类别数
            通过random.choices生成图片索引indices从而进行采样
            """
            if RANK in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            # 如果是DDP模式,则广播采样策略
            if RANK != -1:
                indices = (torch.tensor(dataset.indices) if RANK == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                # 广播索引到其他group
                if RANK != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders
        # 初始化训练时打印的平均损失信息
        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            # DDP模式下打乱数据, ddp.sampler的随机采样数据是基于epoch+seed作为随机种子，
            # 每次epoch不同，随机种子就不同
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            # tqdm 创建进度条，方便训练时 信息的展示
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            # 计算迭代的次数iteration
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            """
            预热训练(前nw次迭代)
            在前nw次迭代中，根据以下方式选取accumulate和学习率
            """
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
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
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            # 混合精度
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                # 计算损失，包括分类损失，objectness损失，框的回归损失
                # loss为总损失值，loss_items为一个元组，包含分类损失，objectness损失，框的回归损失和总损失
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    # 平均不同gpu之间的梯度
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode

                # 如果使用collate_fn4函数来加载dataloader的话, loss *= 4,
                # collate_fn4是有0.5的概率将一个batch里每4张图片拼接在一起作为一张大图训练
                # 还有0.5的概率将一张图放大一倍作为大图训练
                # 具体可看datasets.py中的collate_fn4注释
                if opt.quad:
                    loss *= 4.

            # Backward
            # 反向传播
            scaler.scale(loss).backward()

            # Optimize
            # 模型反向传播accumulate次之后再根据累积的梯度更新一次参数
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log 日志信息打印
            # 打印显存，进行的轮次，损失，target的数量和图片的size等信息
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                # 进度条显示以上信息
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.on_train_batch_end(ni, model, imgs, targets, paths, plots)
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        # 进行学习率衰减
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in [-1, 0]:
            # mAP
            callbacks.on_train_epoch_end(epoch=epoch)
            # 更新EMA的属性
            # 添加include的属性
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            # 判断该epoch是否为最后一轮
            final_epoch = epoch + 1 == epochs
            # 对测试集进行测试，计算mAP等指标
            # 测试时使用的是EMA模型
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = run(data_dict,
                                           batch_size=batch_size // WORLD_SIZE * 2,
                                           imgsz=imgsz,
                                           model=ema.ema,
                                           single_cls=single_cls,
                                           dataloader=val_loader,
                                           save_dir=save_dir,
                                           save_json=is_coco and final_epoch,
                                           verbose=nc < 50 and final_epoch,
                                           plots=plots and final_epoch,
                                           callbacks=callbacks,
                                           compute_loss=compute_loss)

            # Update best mAP
            # 更新best_fitness
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.on_fit_epoch_end(log_vals, epoch, best_fitness, fi)

            """
            保存模型，这里是model与ema都保存了的，还保存了epoch，results，optimizer等信息，
            """
            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
                callbacks.on_model_save(last, epoch, final_epoch, best_fitness, fi)

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in [-1, 0]:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        if not evolve:
            # 如果是coco数据集则单独再测试一次
            if is_coco:  # COCO dataset
                for m in [last, best] if best.exists() else [last]:  # speed, mAP tests
                    results, _, _ = val.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            model=attempt_load(m, device).half(),
                                            iou_thres=0.7,  # NMS IoU threshold for best pycocotools results
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            save_json=True,
                                            plots=False)
            # Strip optimizers
            """
            模型训练完后，strip_optimizer函数将除了模型model或者ema之外的所有东西去除；
            并且对模型进行model.half(), 将Float32的模型->Float16，
            可以减少模型大小，提高inference速度
            """
            for f in last, best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers
        callbacks.on_train_end(last, best, plots, epoch)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    # 释放显存
    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    """
     opt参数解析：
     cfg:                               模型配置文件，网络结构
     data:                              数据集配置文件，数据集路径，类名等
     hyp:                               超参数文件
     epochs:                            训练总轮次
     batch-size:                        批次大小
     img-size:                          输入图片分辨率大小
     rect:                              是否采用矩形训练，默认False
     resume:                            接着打断训练上次的结果接着训练
     nosave:                            不保存模型，默认False
     notest:                            不进行test，默认False
     noautoanchor:                      不自动调整anchor，默认False
     evolve:                            是否进行超参数进化，默认False
     bucket:                            谷歌云盘bucket，一般不会用到
     cache-images:                      是否提前缓存图片到内存，以加快训练速度，默认False
     weights:                           加载的权重文件
     name:                              数据集名字，如果设置：results.txt to results_name.txt，默认无
     device:                            训练的设备，cpu；0(表示一个gpu设备cuda:0)；0,1,2,3(多个gpu设备)
     multi-scale:                       是否进行多尺度训练，默认False
     single-cls:                        数据集是否只有一个类别，默认False
     adam:                              是否使用adam优化器
     sync-bn:                           是否使用跨卡同步BN,在DDP模式使用
     local_rank:                        gpu编号
     logdir:                            存放日志的目录
     workers:                           dataloader的最大worker数量
     """

    parser = argparse.ArgumentParser()
    # 预训练权重文件
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    # 训练模型
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    # 训练路径，包括训练集，验证集，测试集的路径，类别总数等
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    # 使用的超参数文件
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    # 训练的批次
    parser.add_argument('--epochs', type=int, default=300)
    # 训练的每批的图片数量
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    # 图片的分辨率（长宽）
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    # rect-->是否采用矩形训练。矩形推理：比正方形推理减少了更多的冗余部分，满足32的倍数。
    # 矩形训练：将比例相近的图片放在一个batch（由于batch里面的图片shape是一样的）
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    # 接着打断训练上次的结果接着训练，暂时不建议打断训练再resume
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # 不保存模型，只保存最后的检查点
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # 不进行验证，只验证最后一批
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    # 不自动调整anchor，直接使用配置文件里的默认的anchor
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    # 超参数的进化
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    # 谷歌云盘bucket，一般不会用到
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    # 是否提前缓存图片到内存，以加快训练速度
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    # --image-weights：从训练集中采集图像，这些图像由上一个epoch的测试中的mAP反过来加权到图像中（而不是像正常训练那样统一采样图像）。
    # 这将导致在训练期间低mAP而包含内容高的图像被选取的可能性更高
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    # 训练的设备 GPU/CPU
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 是否进行多尺度训练
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    # 数据集是否只有一个类别，默认False
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    # 是否使用adam优化器，默认为False，即SGD
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    # 是否使用跨卡同步BN,在DDP模式使用
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # dataloader的最大worker数量 （使用多线程加载图片）
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    # 训练结果的保存路径
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    # W&B实体
    parser.add_argument('--entity', default=None, help='W&B entity')
    # 训练结果的文件名称
    parser.add_argument('--name', default='exp', help='save to project/name')
    # 创建文件夹需要的参数，用于确定文件夹是否存在。为True时，只有在目录不存在时才创建，已存在不会抛出异常
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    # 线性学习率
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    # 标签平滑
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    # 将数据集加载为W&B文件的表
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    # 日志模型
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    # 要使用的数据集的版本
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    # GPU编号，DDP参数，请勿修改
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    # 要冻结的层数
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


'''
main()函数
'''
def main(opt):
    # Checks
    # 以下使用的函数为utils/general.py文件内定义的
    # 初始化logging
    set_logging(RANK)
    if RANK in [-1, 0]:
        print(colorstr('train: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
        check_git_status()
        check_requirements(requirements=FILE.parent / 'requirements.txt', exclude=['thop'])

    # Resume
    # 是否接着打断上次的结果接着训练
    # check_wandb_resume（utils/loggers/wandb/wandb_utils.py）
    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:  # resume an interrupted run
        # 如果resume是str,则表示传入的是模型的路径地址
        # 否则get_latest_run()函数获取runs文件夹中最近的last.pt
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        # os.path.isfile() 用于判断某一对象(需提供绝对路径)是否为文件
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        # opt参数也全部替换
        # open()函数是打开文件，但文件属于I/O流，需要使用后关闭，每次这样麻烦。
        # 使用with之后可以自动帮我们调用close()方法。此时不必调用f.close()方法。
        # parent获取path的上级路径，parents获取path的所有上级路径。此处获取的是 -->  \runs\train\exp\opt.yaml
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            # 超参数替换
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        # opt.cfg设置为'' 对应着train函数里面的操作(加载权重时是否加载权重里的anchor)
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate # 恢复训练
        LOGGER.info(f'Resuming training from {ckpt}') # 打印从ckpt恢复训练的日志
    else:
        # 检查配置文件信息
        # check_file （utils/general.py）的作用为查找/下载文件 并返回该文件的路径。
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        # 如果模型文件和权重文件为空，弹出警告
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        # 如果要进行超参数进化，重建保存路径
        if opt.evolve:
            opt.project = 'runs/evolve'
            opt.exist_ok = opt.resume
        # increment_path （utils/general.py） 使文件名递增的函数，默认opt.exist_ok为false
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode  -->  支持多机多卡、分布式训练
    # 选择设备
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        from datetime import timedelta
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'
        assert not opt.sync_bn, '--sync-bn known training issue, see https://github.com/ultralytics/yolov5/issues/3998'
        # 根据gpu编号选择设备
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        # 初始化进程组
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=60))

    '''
    训练模式: 如果不进行超参数进化，则直接调用train()函数，开始训练
    '''
    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device)
        if WORLD_SIZE > 1 and RANK == 0:
            _ = [print('Destroying process group... ', end=''), dist.destroy_process_group(), print('Done.')]

    # 进化超参数
    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        # 超参数进化列表,括号里分别为(突变规模, 最小值,最大值)
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
                'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)
        # 加载默认超参数
        with open(opt.hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            # 如果超参数文件中没有'anchors'，则设为3
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        # 使用进化算法时，仅在最后的epoch测试和保存
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')  # download evolve.csv if exists

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
        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                # 选择进化方式
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                # 加载evolve.txt
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                # 选取至多前5次进化的结果
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                # 根据results计算hyp的权重
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                # 根据不同进化方式获得base hyp
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # 超参数进化
                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                # 获取突变初始值
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

            # Constrain to limits
            '''
            修剪hyp在规定范围里
            为了防止突变过程，导致参数出现明显不合理的范围，
            需要用一个范围进行框定，将超出范围的内容剪切掉。
            具体方法如下
            '''
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            # 训练变化
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            """
                写入results和对应的hyp到evolve.txt
                evolve.txt文件每一行为一次进化的结果
                 一行中前七个数字为(P, R, mAP, F1, val_losses=(box, obj, cls))，之后为hyp
                保存hyp到yaml文件
            """
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        # 可视化结果    plot_evolve（utils/plots.py）
        plot_evolve(evolve_csv)
        print(f'Hyperparameter evolution finished\n'
              f"Results saved to {colorstr('bold', save_dir)}\n"
              f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    # 封装train接口
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
