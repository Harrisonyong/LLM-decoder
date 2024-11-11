#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   DecoderLayer.py
@Time    :   2024/08/29 14:21:47
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   None
'''

import torch
from torch import nn
from typing import Optional, Tuple
from transformers.utils import logging
from .configure_wedoctor import WeDoctorConfig
from .attention_wedoctor import WedoctorAttention
from .mlp_wedoctor import WedoctorMLP
from .layernorm_wedoctor import WedoctorRMSNorm

logger = logging.get_logger(__name__)


WEDOCTOR_ATTENTION_CLASSES = {
    "eager": WedoctorAttention
}

class WedoctorDecoderLayer(nn.Module):
    def __init__(self, config: WeDoctorConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = WEDOCTOR_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx) 
        self.mlp = WedoctorMLP(config=config)
        self.input_layernorm = WedoctorRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = WedoctorRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
        hidden_states float tensor: (batch, seq_len, embed_dim)
        attention_mask float tensor: (batch, seq_len)
        output_attentions
        use_cache
        """
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)

        # self_attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_value = past_key_value,
            output_attentions = output_attentions,
            use_cache = use_cache,
            cache_position = cache_position
        )
        
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        
        # fully connected
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
