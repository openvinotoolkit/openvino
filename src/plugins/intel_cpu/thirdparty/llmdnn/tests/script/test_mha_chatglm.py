# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math
import sys
import torch
import numpy as np
import llmdnn as ld
from torch import nn
import torch.nn.functional as F

# copy from chatglm-6b/modeling_chatglm.py
class GPTNeoXAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.norm_factor = torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(torch.get_default_dtype())

    def forward(self,
                query_layer,    # [b, np, sq, hn]
                key_layer,      # [b, np, sk, hn]
                value_layer,    # [b, np, sk, hn]
                attention_mask  # [b, np, s, s]
                ):
        return self.attention_fn(query_layer, key_layer, value_layer, attention_mask)

    def attention_fn(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask
    ):
        query_layer = query_layer / self.norm_factor

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(0), query_layer.size(1), query_layer.size(2), key_layer.size(2))

        # [b, np, sq, hn] -> [b * np, sq, hn]
        query_layer = query_layer.view(output_size[0] * output_size[1], output_size[2], -1)
        # [b, np, sk, hn] -> [b * np, sk, hn]
        key_layer = key_layer.view(output_size[0] * output_size[1], output_size[3], -1)

        matmul_result = torch.zeros(
            1, 1, 1,
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer,                                # [b * np, sq, hn]
            key_layer.transpose(1, 2),                  # [b * np, hn, sk]
            beta=0.0,
            alpha=1.0,
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # if not (attention_mask == 0).all():
        #     # if auto-regressive, skip
        #     attention_scores.masked_fill_(attention_mask, -10000.0)
        attention_scores = attention_scores + attention_mask
        dtype = attention_scores.dtype
        attention_scores = attention_scores.float()

        attention_probs = F.softmax(attention_scores, dim=-1)

        attention_probs = attention_probs.type(dtype)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [b, sk, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(0), value_layer.size(1), query_layer.size(1), value_layer.size(3))

        # [b, np, sk, hn] -> [b * np, sk, hn]
        value_layer = value_layer.view(output_size[0] * output_size[1], value_layer.size(2), output_size[3])

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [b, sq, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # [b, sq, np, hn] --> [b, sq, hp]
        new_context_layer_shape = (output_size[0], output_size[2], output_size[1] * output_size[3])
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class GPTNeoXAttentionExt:
    def __init__(self, num_attention_heads, hidden_size, head_size_aligned, max_position_embeddings, is_int8=False):
        self.mha = ld.mha_gpt()
        num_heads = num_attention_heads
        head_size = hidden_size // num_attention_heads
        max_seq_len = max_position_embeddings

        head_size_aligned = head_size_aligned
        normal_factor = 1.0 / math.sqrt(head_size)
        qkv_precision_name = 's8' if is_int8 else 'bf16'
        dst_precision_name = 's8' if is_int8 else 'bf16'
        self.mha.create(num_heads, head_size, head_size_aligned, normal_factor, qkv_precision_name,
                dst_precision_name, max_seq_len)

    def forward(self, query, key, value, attention_mask, head_size, key_seq_len):
        return self.mha.exec(query, key, value, attention_mask, head_size, key_seq_len)

    def forward_quant(self, query, key, value, attention_mask, q_quant, k_quant, qk_quant, v_quant, requant):
        # q_dequant, k_dequant, v_dequant, qk_quant, std::vector<float>& qkv_quant
        return self.mha.exec_quant(query, key, value, attention_mask, 1.0 / q_quant, 1.0 / k_quant, 1.0 / v_quant, qk_quant, requant)

HEAD_NUM = 32
SIZE_PER_HEAD = 80
SIZE_PER_HEAD_ALIGN = 96
HIDDEN_SIZE = HEAD_NUM * SIZE_PER_HEAD
MAX_POSITION_EMBEDDINGS = 1024 #2048
def get_ref_model():
    class FakeConfig:
        def __init__(self):
            self.num_attention_heads = HEAD_NUM
            self.hidden_size = HIDDEN_SIZE
            self.max_position_embeddings = MAX_POSITION_EMBEDDINGS
    config = FakeConfig()
    ref_net = GPTNeoXAttention(config)
    ref_net = ref_net.to(dtype=torch.bfloat16)
    return ref_net

def test_chatglm_neox():
    inputs = [
        # q, k, v, attn_mask
        # q: [batch, num_heads, query_seq_len, head_size]
        # k: [batch, num_heads, key_seq_len, head_size]
        # v: [batch, num_heads, value_seq_len, head_size]
        # attn: [batch, 1, query_seq_len, key_seq_len]
        (np.random.random(size=[2, HEAD_NUM, 2, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 32, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 32, SIZE_PER_HEAD]).astype(np.float32),
         np.zeros([2, 1, 2, 32], dtype=np.float32)),
        (np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.zeros([2, 1, 200, 200], dtype=np.float32)),
        (np.random.random(size=[2, HEAD_NUM, 1, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.zeros([2, 1, 1, 200], dtype=np.float32)),
    ]
    ref_net = get_ref_model()
    net = GPTNeoXAttentionExt(HEAD_NUM, HIDDEN_SIZE, SIZE_PER_HEAD_ALIGN, MAX_POSITION_EMBEDDINGS)
    with torch.cpu.amp.autocast():
        for (i, input) in enumerate(inputs):
            q, k, v, attn_mask = input
            q = torch.from_numpy(q).to(torch.bfloat16)
            k = torch.from_numpy(k).to(torch.bfloat16)
            v = torch.from_numpy(v).to(torch.bfloat16)
            attn_mask = torch.from_numpy(attn_mask)
            attn_mask[:,:,:,-2:] = torch.finfo(torch.float32).min
            ref_output = ref_net.forward(q, k, v, attn_mask)

            shape = list(k.shape)
            shape[-2] = MAX_POSITION_EMBEDDINGS
            shape[-1] = SIZE_PER_HEAD_ALIGN
            key_padded = torch.zeros(shape, dtype=torch.bfloat16)
            value_padded = torch.zeros(shape, dtype=torch.bfloat16)
            query_shape = list(q.shape)
            query_shape[-1] = SIZE_PER_HEAD_ALIGN
            query_padded = torch.zeros(query_shape, dtype=torch.bfloat16)
            key_padded[:,:,:k.shape[-2],:k.shape[-1]] = k
            value_padded[:,:,:k.shape[-2],:k.shape[-1]] = v
            query_padded[:,:,:,:q.shape[-1]] = q
            
            output = net.forward(query_padded, key_padded, value_padded, attn_mask, k.size(3), k.size(2))
            if not torch.allclose(ref_output, output, rtol=0.001, atol=0.01):
                print(f"error at index {i} ref:\n{ref_output} \ncur:\n {output} ")
                assert(False)

    print('done.')
    return

def todotest_gpt_neox_int8():
    low = -4
    high = 4
    range_ = high - low
    q_quant = 127.0 / high
    qs = [
            np.random.random(size=[2, HEAD_NUM, 900, SIZE_PER_HEAD]).astype(np.float32)*range_+low,
            np.random.random(size=[2, HEAD_NUM , 1, SIZE_PER_HEAD]).astype(np.float32)*range_+low,
            np.random.random(size=[2, HEAD_NUM , 1, SIZE_PER_HEAD]).astype(np.float32)*range_+low,
        ]
    low = -2
    high = 2
    range_ = high - low
    k_quant = 127.0 / high
    ks = [
            np.random.random(size=[2, HEAD_NUM, 900, SIZE_PER_HEAD]).astype(np.float32)*range_+low,
            np.random.random(size=[2, HEAD_NUM, 901, SIZE_PER_HEAD]).astype(np.float32)*range_+low,
            np.random.random(size=[2, HEAD_NUM, 902, SIZE_PER_HEAD]).astype(np.float32)*range_+low,
        ]
    low = -8
    high = 8
    range_ = high - low
    v_quant = 127.0 / high
    vs = [
            np.random.random(size=[2, HEAD_NUM, 900, SIZE_PER_HEAD]).astype(np.float32)*range_+low,
            np.random.random(size=[2, HEAD_NUM, 901, SIZE_PER_HEAD]).astype(np.float32)*range_+low,
            np.random.random(size=[2, HEAD_NUM, 902, SIZE_PER_HEAD]).astype(np.float32)*range_+low,
        ]
    # q, k, v, attn_mask
    # q: [batch, num_heads, query_seq_len, head_size]
    # k: [batch, num_heads, key_seq_len, head_size]
    # v: [batch, num_heads, value_seq_len, head_size]
    # attn: [1, MAX_POSITION_EMBEDDINGS]
    inputs = []
    for i in range(len(qs)):
        inputs.append((qs[i], ks[i], vs[i], np.zeros([1, ks[i].shape[-2]], dtype=np.float32)))
    
    ref_net = get_ref_model()
    net = GPTNeoXAttentionExt(HEAD_NUM, HIDDEN_SIZE, MAX_POSITION_EMBEDDINGS, True)
    qk_quant, requant = 255.0, 10.0
    with torch.cpu.amp.autocast():
        for (i, input) in enumerate(inputs):
            q, k, v, attn_mask = input
            q = torch.from_numpy(q)
            k = torch.from_numpy(k)
            v = torch.from_numpy(v)
            attn_mask = torch.from_numpy(attn_mask)
            q = (q * q_quant).round().clamp(-128, 127).to(torch.int8)
            k = (k * k_quant).round().clamp(-128, 127).to(torch.int8)
            v = (v * v_quant).round().clamp(-128, 127).to(torch.int8)
            ref_output = ref_net.forward(q, k, v, attn_mask, q_quant, k_quant, qk_quant, v_quant, requant)
            output = net.forward_quant(q, k, v, attn_mask, q_quant, k_quant, qk_quant, v_quant, [requant,])
            if (torch.abs(ref_output- output) > 2).any():
                print(f"error at index {i} ref:\n{ref_output} \ncur:\n {output} ")
                assert(False)

    print('done.')
    return

if __name__ == "__main__":
    test_chatglm_neox()