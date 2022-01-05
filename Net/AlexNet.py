# -*- coding: utf-8 -*-
"""
# @file name  : lenet.py
# @author     : yts3221@126.com
# @date       : 2019-08-21 10:08:00
# @brief      : lenet模型定义
"""
import torch.nn as nn
import torch.nn.functional as F


class AlexNet_bn(nn.Module):
    def __init__(self, classes):
        """
        该模型的输入尺寸是(3*224*224)
        :param classes:
        """
        super(AlexNet_bn, self).__init__()
        self.feature_extraction = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # 第二层卷积
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # 第三层卷积
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # 第四层卷积
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # 第五层卷积
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 5 * 5, out_features=3200),
            nn.BatchNorm1d(3200),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=3200, out_features=3200, bias=True),
            nn.BatchNorm1d(3200),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=3200, out_features=classes),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        # x = x.reshape(-1, 256 * 2 * 2)
        x = x.view(x.size(0), -1)
        # x = x.view(-1, 256 * 5 * 5, 1, 1)
        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 1)
                m.bias.data.zero_()


class AlexNet(nn.Module):
    def __init__(self, classes):
        """
        该模型的输入尺寸是(3*224*224)，没有部分bn
        :param classes:
        """
        super(AlexNet, self).__init__()
        self.feature_extraction = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # 第二层卷积
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # 第三层卷积
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # 第四层卷积
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # 第五层卷积
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 5 * 5, out_features=3200),
            # nn.BatchNorm1d(3200),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=3200, out_features=3200, bias=True),
            # nn.BatchNorm1d(3200),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=3200, out_features=classes),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        # x = x.reshape(-1, 256 * 2 * 2)
        x = x.view(x.size(0), -1)
        # x = x.view(-1, 256 * 5 * 5, 1, 1)
        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 1)
                m.bias.data.zero_()
