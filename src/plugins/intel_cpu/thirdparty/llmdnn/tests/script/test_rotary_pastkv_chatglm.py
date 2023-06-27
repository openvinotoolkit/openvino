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
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

# copy from chatglm-6b/modeling_chatglm.py
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000, precision=torch.half, learnable=False):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        #inv_freq = inv_freq.half()
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
            # use f32 to pass accuracy test
            # Build here to make `torch.jit.trace` work.
            self.max_seq_len_cached = max_position_embeddings
            t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[:, None, :]
            self.sin_cached = emb.sin()[:, None, :]
            
        self.precision = precision

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        pass

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def _apply(self, fn):
        if self.cos_cached is not None:
            self.cos_cached = fn(self.cos_cached)
        if self.sin_cached is not None:
            self.sin_cached = fn(self.sin_cached)
        return super()._apply(fn)


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # position_id: [b, sq], q, k: [b, sq, np, hn], cos: [sq, 1, hn] -> [b, sq, 1, hn]
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
        F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k

class SelfAttention(torch.nn.Module):
    def __init__(self, max_position_embeddings, hidden_size, num_attention_heads,
                 layer_id, hidden_size_per_attention_head=None, bias=True,
                 params_dtype=torch.float, position_encoding_2d=True, empty_init=True):
        super(SelfAttention, self).__init__()

        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.hidden_size_per_partition = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_heads_per_partition = num_attention_heads
        self.position_encoding_2d = position_encoding_2d
        self.rotary_emb = RotaryEmbedding(
            self.hidden_size // (self.num_attention_heads * 2)
            if position_encoding_2d
            else self.hidden_size // self.num_attention_heads,
            max_position_embeddings,
            base=10000,
            precision=torch.half,
            learnable=False,
        )

        self.scale_mask_softmax = None

        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = hidden_size // num_attention_heads
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head

        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head

    @staticmethod
    def attention_mask_func(attention_scores, attention_mask):
        attention_scores.masked_fill_(attention_mask, -10000.0)
        return attention_scores

    def split_tensor_along_last_dim(self, tensor, num_partitions,
                                    contiguous_split_chunks=False):
        """Split a tensor along its last dimension.
        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                    in memory.
        """
        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)

        return tensor_list

    def attention_fn(
            self,
            query_layer,
            key_layer,
            value_layer,
            layer_past=None
    ):
        # batch, seqlen, num_attention_heads, hidden_size_per_attention_head
        b, seq_len, nh, hidden_size = key_layer.shape

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(0), query_layer.size(2), query_layer.size(1), key_layer.size(1))

        # # [sq, b, np, hn] -> [sq, b * np, hn]
        # query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # # [sk, b, np, hn] -> [sk, b * np, hn]
        # key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        # [b, sq, np, hn] -> [b, np, sq, hn]
        query_layer = query_layer.transpose(1, 2)
        # [b, sk, np, hn] -> [b, np, sk, hn]
        key_layer = key_layer.transpose(1, 2)
        # [b, np, sq, hn] -> [b * np, sq, hn]
        #query_layer = query_layer.view(output_size[0] * output_size[1], output_size[2], -1)
        # [b, np, sk, hn] -> [b * np, sk, hn]
        #key_layer = key_layer.view(output_size[0] * output_size[1], output_size[3], -1)

        # value_layer -> context layer.
        # [b, sk, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(0), value_layer.size(2), query_layer.size(1), value_layer.size(3))

        # # change view [sk, b * np, hn]
        # value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
        # [b, sk, np, hn] -> [b, np, sk, hn]
        value_layer = value_layer.transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=2)

        # [b, np, sk, hn] -> [b * np, sk, hn]
        #value_layer = value_layer.view(output_size[0] * output_size[1], value_layer.size(2), output_size[3])
        return query_layer, key_layer, value_layer

    def forward(
            self,
            qkv: torch.Tensor,              # [batch, seq_len, 3 * hidden_size]
            position_ids,                   # [batch, 2, query_seq_len]
            layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        """
        hidden_states: [batch, seq_len, hidden_size]
        attention_mask: [(1, 1), seq_len, seq_len]
        """

        # [batch, seq_len, 3 * hidden_size]
        mixed_raw_layer = qkv # self.query_key_value(hidden_states)

        # [batch, seq_len, 3 * hidden_size] --> [batch, seq_len, num_attention_heads, 3 * hidden_size_per_attention_head]
        new_tensor_shape = mixed_raw_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_raw_layer = mixed_raw_layer.view(*new_tensor_shape)

        # [batch, seq_len, num_attention_heads, hidden_size_per_attention_head]
        (query_layer, key_layer, value_layer) = self.split_tensor_along_last_dim(mixed_raw_layer, 3)

        if self.position_encoding_2d:
            q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
            k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
            cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1)
            # xxx
            position_ids, block_position_ids = position_ids[:, 0, :].contiguous(), \
                position_ids[:, 1, :].contiguous()
            q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
            q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
            query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1))
            key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1))
        else:
            position_ids = position_ids.transpose(0, 1)
            cos, sin = self.rotary_emb(value_layer, seq_len=position_ids.max() + 1)
            # [batch, seq_len, num_attention_heads, hidden_size_per_attention_head]
            query_layer, key_layer = apply_rotary_pos_emb_index(query_layer, key_layer, cos, sin, position_ids)

        query_layer = query_layer.to(dtype=torch.bfloat16)
        key_layer = key_layer.to(dtype=torch.bfloat16)
        
        # [batch, seq_len, hidden_size]
        query_layer, key_layer, value_layer = self.attention_fn(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            layer_past=layer_past,
        )

        return query_layer, key_layer, value_layer


class GPTNeoXAttentionExt:
    def __init__(self, num_attention_heads, hidden_size, head_size_aligned, max_position_embeddings, rotary_emb_base, rotary_pct, is_int8=False):
        self.emd = ld.emb_gpt()
        num_heads = num_attention_heads
        head_size = hidden_size // num_attention_heads
        max_seq_len = max_position_embeddings

        qkv_precision_name = 's8' if is_int8 else 'bf16'
        dst_precision_name = 's8' if is_int8 else 'bf16'
        self.emd.create(num_heads, head_size, head_size_aligned, qkv_precision_name,
                dst_precision_name, max_seq_len, rotary_emb_base, rotary_pct, True)

    # qkv: [batch, seq_len, (num_heads * 3 * head_size)]
    # layer_past_padded: [batch, num_attention_heads, MAX_SEQ_LEN, head_size_aligned]
    # past_seq_len: past_seq_len==layer_past.shape[-2]
    # return:
    #       0: (k, v): ([batch, num_attention_heads, MAX_SEQ_LEN, head_size_aligned], [batch, num_attention_heads, MAX_SEQ_LEN, head_size_aligned])
    #       1: query: [batch, num_attention_heads, seq_len, head_size_aligned]
    #       2: k: [batch, num_attention_heads, MAX_SEQ_LEN, head_size_aligned]
    #       3: v: [batch, num_attention_heads, MAX_SEQ_LEN, head_size_aligned]
    def forward(self, qkv, layer_past_key_src, layer_past_value_src, layer_past_key_dst, layer_past_value_dst, query_padded, past_seq_len, position_ids):
        self.emd.exec_position(qkv, layer_past_key_src, layer_past_value_src, layer_past_key_dst, layer_past_value_dst, query_padded, past_seq_len, position_ids)


HEAD_NUM = 32
SIZE_PER_HEAD = 80
SIZE_PER_HEAD_ALIGN = 96
HIDDEN_SIZE = HEAD_NUM * SIZE_PER_HEAD
MAX_POSITION_EMBEDDINGS = 1024 #2048
ROTARY_EMB_BASE = 10000
ROTARY_PCT = 0.5
MAX_SEQ_LEN = 1024
def get_ref_model():
    ref_net = SelfAttention(MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE, HEAD_NUM, 0, None)
    ref_net = ref_net.to(dtype=torch.bfloat16)
    return ref_net

def test_chatglm():
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
    net_seq = GPTNeoXAttentionExt(HEAD_NUM, HIDDEN_SIZE, SIZE_PER_HEAD, MAX_POSITION_EMBEDDINGS, ROTARY_EMB_BASE, ROTARY_PCT)
    with torch.cpu.amp.autocast():
        for (i, input) in enumerate(inputs):
            qkv, layer_past_key, layer_past_value = input
            qkv = torch.from_numpy(qkv).to(torch.bfloat16)
            layer_past_key = torch.from_numpy(layer_past_key).to(torch.bfloat16)
            layer_past_value = torch.from_numpy(layer_past_value).to(torch.bfloat16)
            past_seq_len = layer_past_key.shape[-2]
            shape = list(layer_past_key.shape)

            if qkv.size(1) > 1:
                seq_batch1 = torch.arange(end=qkv.size(1) - 1, dtype=torch.int32)
                seq_batch1 = torch.concat((seq_batch1, seq_batch1[-1:]))
                block_batch1 = torch.concat((torch.zeros(qkv.size(1) -1, dtype=torch.int32), torch.ones(1, dtype=torch.int32)))
            else:
                seq_batch1 = torch.tensor([3], dtype=torch.int32)
                block_batch1 = torch.tensor([5], dtype=torch.int32)
            seq_ids = torch.empty((qkv.size(0), 2, qkv.size(1)), dtype=torch.int32)
            seq_ids[:, 0, :] = seq_batch1
            seq_ids[:, 1, :] = block_batch1
            query_ref, key_ref, value_ref = ref_net.forward(qkv, seq_ids, (layer_past_key, layer_past_value))
            query_ref = query_ref.to(dtype=torch.bfloat16)
            key_ref = key_ref.to(dtype=torch.bfloat16)
            
            # no prealloc past kv
            layer_past_key_seq = torch.zeros(key_ref.shape, dtype=torch.bfloat16)
            layer_past_value_seq = torch.zeros(value_ref.shape, dtype=torch.bfloat16)
            query_seq = torch.zeros(query_ref.shape, dtype=torch.bfloat16)
            net_seq.forward(qkv, layer_past_key, layer_past_value, layer_past_key_seq, layer_past_value_seq, query_seq, past_seq_len, seq_ids)
            query, key, value = query_seq, layer_past_key_seq, layer_past_value_seq
            # check query
            if not torch.allclose(query_ref, query, rtol=0.001, atol=0.01):
                print(f"error at sequence query index {i} ref:\n{query_ref} \ncur:\n {query} ")
                #assert(False)
            # check key
            if not torch.allclose(key_ref, key, rtol=0.001, atol=0.01):
                print(f"error at sequence key index {i} ref:\n{key_ref} \ncur:\n {key} ")
                assert(False)
            # check value
            if not torch.allclose(value_ref, value, rtol=0.001, atol=0.01):
                print(f"error at sequence value index {i} ref:\n{value_ref} \ncur:\n {value} ")
                assert(False)

            # prealloc past kv
            shape[-2] = MAX_SEQ_LEN
            shape[-1] = SIZE_PER_HEAD_ALIGN
            layer_past_key_padded = torch.zeros(shape, dtype=torch.bfloat16)
            layer_past_value_padded = torch.zeros(shape, dtype=torch.bfloat16)
            query_shape = list(shape)
            query_shape[2] = qkv.shape[1]
            query_padded = torch.zeros(query_shape, dtype=torch.bfloat16)
            layer_past_key_padded[:,:,:layer_past_key.shape[-2],:layer_past_key.shape[-1]] = layer_past_key
            layer_past_value_padded[:,:,:layer_past_key.shape[-2],:layer_past_key.shape[-1]] = layer_past_value
            net.forward(qkv, layer_past_key_padded, layer_past_value_padded, layer_past_key_padded, layer_past_value_padded, query_padded, past_seq_len, seq_ids)
            query, key, value = query_padded, layer_past_key_padded, layer_past_value_padded
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
    test_chatglm()