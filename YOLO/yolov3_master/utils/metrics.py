# Model validation metrics

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import general


def fitness(x):
    """根据模型指标P, R, mAP@0.5, mAP@0.5:0.95，返回一个综合分数,用来选取best.pt"""
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """计算每个类别的ap
    :param tp:根据iou阈值计算的true positive, ndarray, [n, 10],
                10表示range[0.5, 0.95],间隔0.05取一个iou阈值,预测与标签超过这个iou阈值才为tp
    :param conf:置信度，ndarray, [n, 1]
    :param pred_cls:预测类别，ndarray, [n, 1]
    :param plot:是否画mAP@0.5的PR曲线
    """
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    # 将tp，conf，pred_cls按照置信度从大到小排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    # 将target_cls去重，获得类别
    unique_classes = np.unique(target_cls)
    # 获得类别数
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    # 初始化坐标x,y
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    # 初始化指标，ap，precision，recall
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))

    # 对每个类别处理
    for ci, c in enumerate(unique_classes):
        # 选取类别为c的索引
        i = pred_cls == c
        # c类别标签的数量
        n_l = (target_cls == c).sum()  # number of labels
        # c类别预测的数量
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            # 累计计算fp，tp
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            # 计算recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            # 计算precision
            precision = tpc / (tpc + fpc)  # precision curve
            # 插值，方便绘制基于iou_thres=0.5的准确率曲线
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            # 根据precision与recall计算ap
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    # 根据precision与recall计算f1值
    f1 = 2 * p * r / (p + r + 1e-16)
    # 画PR曲线，F1曲线，Precision, recall曲线(后三个的横坐标x为置信度)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):
    """根据precision与recall计算ap, 计算PR曲线下的面积"""
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    # np.maximum.accumulate 计算数组的累计最大值
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        # np.trapz求积分, 求得PR曲线下的面积
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    """计算预测的混淆矩阵"""
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        # 筛选大于置信度阈值的预测框
        detections = detections[detections[:, 4] > self.conf]
        # 标签类别
        gt_classes = labels[:, 0].int()
        # 预测类别
        detection_classes = detections[:, 5].int()
        # 标签框与预测框的iou，(M, N)
        iou = general.box_iou(labels[:, 1:], detections[:, :4])

        # 找到大于iou阈值的, x是一个包含两个元素的元组，
        # 第一个是满足条件的行索引, 第二个是满足条件的列索引，长度为iou中大于iou阈值的个数n
        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            # torch.stack(x, 1).shape (n, 2), iou[x[0], x[1]][:, None].shape (n, 1)
            # matches.shape (n, 3), 前两列为满足iou阈值的行索引和列索引, 第三列是该iou值
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                """因为np.unique去重机制是取先见到的元素，之后的重复元素去掉，
                所以总的来说下面就是对每个标签框选其iou最大的预测框作为匹配
                每个预测框也是选其iou最大的标签框作为匹配，留下的match都是相互iou最大的
                """
                # matches按iou大小 从大到小排序
                matches = matches[matches[:, 2].argsort()[::-1]]
                # 按照列索引去重之后的matches，
                # np.unique(matches[:, 1], return_index=True)[1]返回的是列索引去重的索引
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches按iou大小 从大到小排序
                matches = matches[matches[:, 2].argsort()[::-1]]
                # 按照行索引去重之后的matches，
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        # m0, m1为行索引和列索引
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            # 选取该match的标签框类别为gc的
            j = m0 == i
            # 如果有m0中有索引i的对应，则对应混淆矩阵位置的值+1
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            # 否则就是目标漏检+1, 也就是将目标错误检测成了背景, 背景的误检
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                # 如果m1中没有索引i的对应，则目标误检+1, 也就是将背景错误检测成了目标, 背景的漏检
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, save_dir='', names=()):
        """绘制混淆矩阵"""
        try:
            import seaborn as sn
            # 以概率的形式显示矩阵
            array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            pass

    def print(self):
        """打印矩阵"""
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    """画PR曲线"""
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        # 仅画PR曲线
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    # 画所有类别的综合PR曲线, 并显示map
    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    """以置信度为横坐标画曲线"""
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
