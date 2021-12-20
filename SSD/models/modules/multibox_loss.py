# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from SSD_pytorch.utils.config import opt
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    SSD的损失函数，继承nn.Module，定义为一个网络模型
    Compute Targets:
    计算标准
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
           通过匹配 真值框 与 预测框 的IOU重叠率 来产生分类误差
           默认IOU>0.5即为正样本
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
           通过 编码 真值框与 对应匹配的预测框之间偏移的方差  来产生定位回归误差
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
           硬性负开采   默认负样本：正样本为3:1 参考:https://blog.csdn.net/u012285175/article/details/77866878

    Objective Loss:
        总损失
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Lconf是通过交叉熵计算。Lloc是SmoothL1损失。 α一般为1
        Args:
            c: class confidences,分类置信度
            l: predicted boxes,预测框
            g: ground truth boxes 真值框
            N: number of matched default boxes 匹配到真值框的正样本预测框总数
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu  #true
        self.num_classes = num_classes  #  分类数  20类+1背景
        self.threshold = overlap_thresh  #IOU阈值0.5 （>0.5即为正样本）
        self.background_label = bkg_label  # 0：背景标签
        self.encode_target = encode_target  #false  编码标签
        self.use_prior_for_matching = prior_for_matching   #true  使用先验匹配
        self.do_neg_mining = neg_mining  #true 使用硬性负开采
        self.negpos_ratio = neg_pos  # 3  负比率，即硬性负开采中 负样本：正样本为3:1
        self.neg_overlap = neg_overlap  # 0.5
        self.variance = opt.voc['variance']   #方差

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
             预测框（元组）：包含 定位预测、分类预测 和  priors boxes（不同feature map生成的锚结果）
                   内容包含：
                   loc定位  (batch_size,num_priors,4)
                   conf分类  (batch_size,num_priors,num_classes)
                   priors boxes   (num_priors,4)

                    # loc_data  通过网络输出的定位的预测 [32,8732,4]
                    # conf_data  通过网络输出的分类的预测 [32,8732,21]
                    # priors 不同feature map根据公式生成的锚结果  [8732,4]  一张图片总共产生8732个框
                    #之所以称为锚，而不叫预测框。是因为锚是通过公式生成的，而不是通过网络预测出来的。

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
            目标真值：数据集中对于一个batch的真值框和类别
                      [32,num_objs,5]   32：batch大小， num_objs ：一张图片的物体数，5：前四个数为坐标，最后一个数为类别
        """

        # 损失的计算  是通过 网络输出值与 事先生成锚的结果 进行计算
        loc_data, conf_data, priors = predictions
        # batch数  32
        num = loc_data.size(0)
        # [8732,4] 不同feature map根据公式生成的锚结果
        priors = priors[:loc_data.size(1), :]
        # 8732  总共的预测框数目
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # 匹配 不同feature map生成的锚结果  与  真值框,将结果保存
        # 新建tensor  定位[32,8732,4]
        loc_t = torch.Tensor(num, num_priors, 4)
        # 新建tensor  分类[32,8732]
        conf_t = torch.LongTensor(num, num_priors)

        # 遍历batch中的每一张照片  num=32
        # 使用 真值框 与 所有feature map的每个网格生成的共8732个锚 进行匹配
        for idx in range(num):
            #[2,4] 该图片有两个物体，值为每个物体的4个坐标
            #truths为一张图片中 两个物体对应的真值框坐标
            truths = targets[idx][:, :-1].data
            # 2  该图片有两个物体，最后一个值为每个物体的类别
            #labels为一张图片中 两个物体对应的类别真值
            labels = targets[idx][:, -1].data
            # [8732,4]  defaults  不同feature map根据公式生成的锚结果
            defaults = priors.data
            # 通过匹配 真值框 与 锚 的IOU重叠率来进行选择。将结果存入新建的tensor  loc_t、conf_t中
            # threshold 阈值
            # loc_t[8732,4], conf_t[8732]保存一个batch中的坐标偏移值  和 分类误差
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        # loc_t, conf_t保存一个batch中的坐标偏移值  和 分类误差
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        # pos [32,8732]  由0,1组成，匹配到真实框的位置置为1
        pos = conf_t > 0
        # [32,1] 每个数代表 一张图片上匹配到的锚数目
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        # pos_idx [32,8732,4]   由0,1组成  匹配到真实框的位置置为1
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # loc_p [461,4]   从网络输出的回归框中 拿到 匹配到真值框 对应的定位坐标
        loc_p = loc_data[pos_idx].view(-1, 4)
        # loc_t [461,4]
        loc_t = loc_t[pos_idx].view(-1, 4)
        # 计算算回归定位损失   loc_p 通过网络输出的预测框   loc_t 通过与真值框匹配的锚
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)



        #论文中先将每一个物体位置上对应 predictions（default boxes）是 negative 的 boxes 进行排序，
        # 按照 default boxes 的 confidence 的大小。 选择最高的几个，保证最后 negatives、positives 的比例在 3:1

        # 对于硬性负开采 在batch计算负样本的最大置信度
        # conf_data:[32,9732,21]    batch_conf:[279424,21]  行为网络预测的所有分类  列为网络预测的类别数
        batch_conf = conf_data.view(-1, self.num_classes)
        # batch_conf.gather 取出batch_conf中 匹配的锚对应的预测分类值
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        # 硬性负开采
        # loss_c[pos] = 0  # filter out pos boxes for now  过滤掉正样本pos box
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now  过滤掉正样本pos box
        # 按每行降序排序  loss_idx为排序后的序号
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # 统计 正样本数目（匹配到真值框的预测框总数  只有匹配到才能计算损失）
        num_pos = pos.long().sum(1, keepdim=True)
        # 统计 负样本数目
        # torch.clamp(input, min, max, out=None)  将输入张量 input 所有元素限制在区间 [min, max] 中并返回一个结果张量.
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        # 分类损失 包含正样本和负样本   conf_data：通过网络输出的分类的预测 [32,8732,21]
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        # conf_t 保存分类误差
        targets_weighted = conf_t[(pos+neg).gt(0)]
        # 分类损失使用交叉熵    size_average ：如果为TRUE，loss则是平均值，需要除以输入 tensor 中 element 的数目
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        # 与真值框匹配到的预测框总数
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        # 定位损失、分类损失
        return loss_l, loss_c
