#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Project   : Machine-Learning
# File      : KNN_test.py
# Author    : GXL
# Date      : 2019/7/29

import numpy as np
import collections


def createDateSet():
    # 分类坐标，二维数组（特征）
    group = np.array([[11, 13], [1, 145], [3, 120], [14, 20]])
    # （标签）
    lables = np.array(["爱情片", "动作片", "恐怖片", "喜剧片"])
    return group, lables


def classify0(inX, dataSet, lables, k):
    """
    函数说明:kNN算法,分类器
    :param inX:用于分类的数据(测试集)
    :param dataSet:用于训练的数据(训练集)
    :param lables:分类标签
    :param k:kNN算法参数,选择距离最小的k个点
    """
    # 计算距离
    # 如果axis取None，即将数组/矩阵中的元素全部加起来，得到一个和。
    # axis=1为横向相加，=0为纵向相加
    dist = np.sum((inX - dataSet) ** 2, axis=1) ** 0.5
    print("dist:\t", dist)
    # k个最近的标签
    # argsort()函数，是numpy库中的函数，排序后获取坐标
    lables_k = [lables[index] for index in np.argsort(dist)[:k]]
    print("lables:\t", lables_k)
    # 出现次数最多的标签即为最终类别
    # most_common(指定一个参数n，列出前n个元素，不指定参数，则列出所有)
    label = collections.Counter(lables_k).most_common(1)[0][0]
    print("tuple:", collections.Counter(lables_k).most_common(1))
    return label


if __name__ == '__main__':
    group, lables = createDateSet()
    test_ = [0, 0]
    lable = classify0(test_, group, lables, 3)
    print(lable)
