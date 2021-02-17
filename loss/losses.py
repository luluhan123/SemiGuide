#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-02-15 17:13:08
# @Author  : lulu.han (lulu.han@aliyun.com)
# @Link    : https://github.com/luluhan123
# @Version : $Id$

import os
import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss
