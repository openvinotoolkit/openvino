# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math
import sys
import torch
import numpy as np
import llmdnn as ld
from torch import nn

# copy from transformers/models/gpt_neox/modeling_gpt_neox.py
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached[:seq_len, ...].to(x.device), self.sin_cached[:seq_len, ...].to(x.device)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[..., offset : q.shape[-2] + offset, :]
    sin = sin[..., offset : q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GPTNeoXAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims, config.max_position_embeddings, base=config.rotary_emb_base
        )

    # qkv: [batch, seq_len, (num_heads * 3 * head_size)]
    # layer_past: [batch, num_attention_heads, past_seq_len, head_size]
    # return: (key, value), key/value: [batch, num_attention_heads, seq_len+layer_past[0].shape[-2], head_size]
    def forward(self, qkv, layer_past):
        has_layer_past = layer_past is not None
        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[-2]
            seq_len += offset
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, offset=offset)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value)

        return present, query, key, value

class GPTNeoXAttentionExt:
    def __init__(self, num_attention_heads, hidden_size, head_size_aligned, max_position_embeddings, rotary_emb_base, rotary_pct, is_int8=False):
        self.emd = ld.emb_gpt()
        num_heads = num_attention_heads
        head_size = hidden_size // num_attention_heads
        max_seq_len = max_position_embeddings

        qkv_precision_name = 's8' if is_int8 else 'bf16'
        dst_precision_name = 's8' if is_int8 else 'bf16'
        self.emd.create(num_heads, head_size, head_size_aligned, qkv_precision_name,
                dst_precision_name, max_seq_len, rotary_emb_base, rotary_pct)

    # qkv: [batch, seq_len, (num_heads * 3 * head_size)]
    # layer_past_padded: [batch, num_attention_heads, MAX_SEQ_LEN, head_size_aligned]
    # past_seq_len: past_seq_len==layer_past.shape[-2]
    # return:
    #       0: (k, v): ([batch, num_attention_heads, MAX_SEQ_LEN, head_size_aligned], [batch, num_attention_heads, MAX_SEQ_LEN, head_size_aligned])
    #       1: query: [batch, num_attention_heads, seq_len, head_size_aligned]
    #       2: k: [batch, num_attention_heads, MAX_SEQ_LEN, head_size_aligned]
    #       3: v: [batch, num_attention_heads, MAX_SEQ_LEN, head_size_aligned]
    def forward(self, qkv, layer_past_key_padded, layer_past_value_padded, query_padded, past_seq_len):
        self.emd.exec(qkv, layer_past_key_padded, layer_past_value_padded, query_padded, past_seq_len)


HEAD_NUM = 32
SIZE_PER_HEAD = 80
SIZE_PER_HEAD_ALIGN = 96
HIDDEN_SIZE = HEAD_NUM * SIZE_PER_HEAD
MAX_POSITION_EMBEDDINGS = 1024 #2048
ROTARY_EMB_BASE = 10000
ROTARY_PCT = 0.25
MAX_SEQ_LEN = 1024
def get_ref_model():
    class FakeConfig:
        def __init__(self):
            self.num_attention_heads = HEAD_NUM
            self.hidden_size = HIDDEN_SIZE
            self.rotary_pct = ROTARY_PCT
            self.max_position_embeddings = MAX_POSITION_EMBEDDINGS
            self.rotary_emb_base = ROTARY_EMB_BASE
    config = FakeConfig()
    ref_net = GPTNeoXAttention(config)
    ref_net = ref_net.to(dtype=torch.bfloat16)
    return ref_net

def test_gpt_neox():
    inputs = [
        # qkv: [batch, seq_len, (num_heads * 3 * head_size)]
        # layer_past: [batch, num_attention_heads, past_seq_len, head_size]
        (np.random.random(size=[2, 200, 3 * HEAD_NUM * SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 0, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 0, SIZE_PER_HEAD]).astype(np.float32)),
        (np.random.random(size=[2, 1, 3 * HEAD_NUM * SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32)),
    ]
    ref_net = get_ref_model()
    net = GPTNeoXAttentionExt(HEAD_NUM, HIDDEN_SIZE, SIZE_PER_HEAD_ALIGN, MAX_POSITION_EMBEDDINGS, ROTARY_EMB_BASE, ROTARY_PCT)
    with torch.cpu.amp.autocast():
        for (i, input) in enumerate(inputs):
            qkv, layer_past_key, layer_past_value = input
            qkv = torch.from_numpy(qkv).to(torch.bfloat16)
            past_seq_len = layer_past_key.shape[-2]
            shape = list(layer_past_key.shape)
            shape[-2] = MAX_SEQ_LEN
            shape[-1] = SIZE_PER_HEAD_ALIGN
            layer_past_key_padded = torch.zeros(shape, dtype=torch.bfloat16)
            layer_past_value_padded = torch.zeros(shape, dtype=torch.bfloat16)
            query_shape = list(shape)
            query_shape[2] = qkv.shape[1]
            query_padded = torch.zeros(query_shape, dtype=torch.bfloat16)
            layer_past_key = torch.from_numpy(layer_past_key).to(torch.bfloat16)
            layer_past_value = torch.from_numpy(layer_past_value).to(torch.bfloat16)
            layer_past_key_padded[:,:,:layer_past_key.shape[-2],:layer_past_key.shape[-1]] = layer_past_key
            layer_past_value_padded[:,:,:layer_past_key.shape[-2],:layer_past_key.shape[-1]] = layer_past_value
            ref_output, query_ref, key_ref, value_ref = ref_net.forward(qkv, (layer_past_key, layer_past_value))
            query_ref = query_ref.to(dtype=torch.bfloat16)
            key_ref = key_ref.to(dtype=torch.bfloat16)
            net.forward(qkv, layer_past_key_padded, layer_past_value_padded, query_padded, past_seq_len)
            output, query, key, value = (layer_past_key_padded, layer_past_value_padded), query_padded, layer_past_key_padded, layer_past_value_padded
            # check output
            if not torch.allclose(ref_output[0].to(dtype=torch.bfloat16), output[0][:,:,:ref_output[0].shape[-2],:ref_output[0].shape[-1]], rtol=0.001, atol=0.01):
                print(f"error at past key index {i} ref:\n{ref_output[0]} \ncur:\n {output[0]} ")
                assert(False)
            if not torch.allclose(ref_output[1], output[1][:,:,:ref_output[1].shape[-2],:ref_output[1].shape[-1]], rtol=0.001, atol=0.01):
                print(f"error at past value index {i} ref:\n{ref_output[1]} \ncur:\n {output[1]} ")
                assert(False)
            # check query
            if not torch.allclose(query_ref, query[:,:,:,:query_ref.shape[-1]], rtol=0.001, atol=0.01):
                print(f"error at query index {i} ref:\n{query_ref} \ncur:\n {query} ")
                assert(False)
            # check key
            if not torch.allclose(key_ref, key[:,:,:key_ref.shape[-2],:key_ref.shape[-1]], rtol=0.001, atol=0.01):
                print(f"error at key index {i} ref:\n{key_ref} \ncur:\n {key} ")
                assert(False)
            # check value
            if not torch.allclose(value_ref, value[:,:,:value_ref.shape[-2],:value_ref.shape[-1]], rtol=0.001, atol=0.01):
                print(f"error at value index {i} ref:\n{value_ref} \ncur:\n {value} ")
                assert(False)

    print('done.')
    return

if __name__ == "__main__":
    test_gpt_neox()