#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   PositionEmbeding.py
@Time    :   2024/08/26 16:28:19
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   None
'''

import torch
from torch import nn

class WedoctorRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # dim维度上的固定值
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
    
    @torch.no_grad()
    def forward(self, x, seq_lens=None):
        device = x.device
        dtype = x.dtype
        if seq_lens > self.max_position_embeddings:
            self.max_seq_len_cached = seq_lens
        dim_ids = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        # freqs:[seq_len, dim/2]，考虑序列索引后，dim上的值
        freqs = torch.outer(dim_ids, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos[:seq_lens].to(dtype), sin[:seq_lens].to(dtype)

def rotate_half(x):
    """
    rotate half hidden dims
    
    Args:
        x input_ids
    Returns:    
        after rotate input
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """变种旋转位置编码，将x的前后两部分分别进行cos和sin的累积后累加"""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed