#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   LayerNorm.py
@Time    :   2024/08/23 15:47:04
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   Pytorch LayerNorm
'''

import torch
from torch import nn
class WedoctorRMSNorm(nn.Module):
    """
    RMSNorm(均方根层归一化): root mean square layer normalization
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # 层归一化可训练参数
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # 张量均方根
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 张量元素/均方根+eps的倒数，归一化
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
