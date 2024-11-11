#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   configure_wedoctor.py
@Time    :   2024/08/26 16:00:18
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   None
'''

from transformers import PretrainedConfig

class WeDoctorConfig(PretrainedConfig):
    model_type = "wedoctor"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    def __init__(self,
                 vocab_size=151936,
                 max_position_embeddings=32768,
                 hidden_size=4096,
                 intermediate_size=22016,
                 num_hidden_layers=32,
                 num_attention_heads=32,
                 num_key_value_heads=32,
                 hidden_act="silu",
                 initializer_range=0.02,
                 rms_norm_eps=1e-6,
                 use_cache=True,
                 tie_word_embeddings=False,
                 rope_theta=10000.0,
                 attention_dropout=0.0,
                 use_sliding_window=False,
                 sliding_window=4096,
                 max_window_layers=28,
                 **kwargs):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs)