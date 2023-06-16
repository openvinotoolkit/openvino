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
class GPTNeoXAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.norm_factor = torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(torch.get_default_dtype())

    def forward(self, query, key, value, attention_mask, q_quant=None, k_quant=None, qk_quant=None, v_quant=None, requant=None):
        if q_quant:
            # quant
            query = query.to(torch.float32)
            key = key.to(torch.float32)
            value = value.to(torch.float32)

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, q_quant, k_quant, qk_quant, v_quant)
        if q_quant:
            attn_output = (attn_output * requant).round().clamp(-128, 127).to(torch.int8)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)

        return attn_output

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        # -> [bs, seq_len, hidden_size]
        return tensor

    def _attn(self, query, key, value, attention_mask, q_quant, k_quant, qk_quant, v_quant):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        if q_quant:
            norm_factor = torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor * (1 / q_quant) * (1 / k_quant)
        else:
            norm_factor = torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=(norm_factor),
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        if q_quant:
            attn_weights = (attn_weights * qk_quant).round().clamp(0, 255).to(torch.uint8).to(torch.float32)        
        attn_output = torch.matmul(attn_weights, value)
        if q_quant:
            attn_output = attn_output * ((1 / qk_quant) * (1 / v_quant))            
        return attn_output, attn_weights

class GPTNeoXAttentionExt:
    def __init__(self, num_attention_heads, hidden_size, max_position_embeddings, is_int8=False):
        self.mha = ld.mha_gpt()
        num_heads = num_attention_heads
        head_size = hidden_size // num_attention_heads
        max_seq_len = max_position_embeddings

        head_size_aligned = head_size
        normal_factor = 1.0 / math.sqrt(head_size)
        qkv_precision_name = 's8' if is_int8 else 'bf16'
        dst_precision_name = 's8' if is_int8 else 'bf16'
        self.mha.create(num_heads, head_size, head_size_aligned, normal_factor, qkv_precision_name,
                dst_precision_name, max_seq_len)

    def forward(self, query, key, value, attention_mask):
        return self.mha.exec(query, key, value, attention_mask)

    def forward_quant(self, query, key, value, attention_mask, q_quant, k_quant, qk_quant, v_quant, requant):
        # q_dequant, k_dequant, v_dequant, qk_quant, std::vector<float>& qkv_quant
        return self.mha.exec_quant(query, key, value, attention_mask, 1.0 / q_quant, 1.0 / k_quant, 1.0 / v_quant, qk_quant, requant)

HEAD_NUM = 32
SIZE_PER_HEAD = 80
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

def test_gpt_neox():
    inputs = [
        # q, k, v, attn_mask
        # q: [batch, num_heads, query_seq_len, head_size]
        # k: [batch, num_heads, key_seq_len, head_size]
        # v: [batch, num_heads, value_seq_len, head_size]
        # attn: [1, MAX_POSITION_EMBEDDINGS]
        (np.random.random(size=[2, HEAD_NUM, 2, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 32, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 32, SIZE_PER_HEAD]).astype(np.float32),
         np.zeros([1, 32], dtype=np.float32)),
        (np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.zeros([1, 200], dtype=np.float32)),
        (np.random.random(size=[2, HEAD_NUM, 1, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.random.random(size=[2, HEAD_NUM, 200, SIZE_PER_HEAD]).astype(np.float32),
         np.zeros([1, 200], dtype=np.float32)),
    ]
    ref_net = get_ref_model()
    net = GPTNeoXAttentionExt(HEAD_NUM, HIDDEN_SIZE, MAX_POSITION_EMBEDDINGS)
    with torch.cpu.amp.autocast():
        for (i, input) in enumerate(inputs):
            q, k, v, attn_mask = input
            q = torch.from_numpy(q).to(torch.bfloat16)
            k = torch.from_numpy(k).to(torch.bfloat16)
            v = torch.from_numpy(v).to(torch.bfloat16)
            attn_mask = torch.from_numpy(attn_mask)
            ref_output = ref_net.forward(q, k, v, attn_mask)
            output = net.forward(q, k, v, attn_mask)
            if not torch.allclose(ref_output, output, rtol=0.001, atol=0.01):
                print(f"error at index {i} ref:\n{ref_output} \ncur:\n {output} ")
                assert(False)

    print('done.')
    return

def test_gpt_neox_int8():
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
    test_gpt_neox_int8()