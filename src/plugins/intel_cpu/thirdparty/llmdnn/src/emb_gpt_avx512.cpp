// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <memory.h>

#include "common/simple_parallel.hpp"
#include "common/utility.hpp"
#include "utility_kernel_avx512.hpp"
#include "transpose_kernel_avx512.hpp"
#include "llm_emb_gpt.hpp"
#include "emb_gpt_avx512.hpp"
#include "rotary_kernel_avx512.hpp"

using namespace ov::cpu;

namespace llmdnn {

struct emb_gpt_impl_avx512 : public emb_gpt::impl {
    bool create(const emb_gpt::create_param& param) override;
    void exec(const emb_gpt::exec_param& param) override;

    void initRotery(size_t max_seq_len);
    void applyRotaryPosEmbMemcpy(uint8_t* q_src, uint8_t* k_src, uint8_t* v_src, uint8_t* q_dst, uint8_t** k_dst, uint8_t** v_dst,
        size_t batch, size_t q_seq_len, size_t past_seq_len);
    void applyRotaryPosEmbMemcpyWithPosition2d(uint8_t* q_src, uint8_t* k_src, uint8_t* v_src, uint8_t* q_dst, uint8_t** k_dst, uint8_t** v_dst,
        size_t batch, size_t q_seq_len, size_t past_seq_len, int* position2d_ids);

    emb_gpt::create_param _create_param;
    size_t _head_num = 32;
    size_t _size_per_head = 80;
    size_t _hidden_size = 32 * 80;
    size_t _rotary_emb_base = 10000;
    float _rotary_pct = 0.25;
    size_t _max_seq_len = 400;
    // aligned to cache line
    size_t _size_per_head_aligned = 80;
    int _rotary_ndims = 0;
    std::shared_ptr<float> _cos_cached;
    std::shared_ptr<float> _sin_cached;
    int64_t _input_type_size = 1;
    int64_t _output_type_size = 1;
    bool _use_position2d = false;
};

bool emb_gpt_impl_avx512::create(const emb_gpt::create_param& param) {
    if (param.qkv_precision != dnnl_bf16) {
        std::cout << "input precision must be bf16 or int8.\n";
        return false;
    }
    // TODO: support s8
    // if (param.dst_precision != dnnl_bf16 && param.dst_precision != dnnl_s8) {
    //     std::cout << "dst precision must be bf16 or int8.\n";
    //     return false;
    // }
    _create_param = param;

    _head_num = param.num_heads;
    _size_per_head = param.head_size;
    _size_per_head_aligned = param.head_size_aligned;
    _hidden_size = param.head_size * param.num_heads;
    _rotary_emb_base = param.rotary_emb_base;
    _rotary_pct = param.rotary_pct;
    _max_seq_len = param.max_seq_len;
    _input_type_size = sizeof(ov::bfloat16);
    _output_type_size = sizeof(ov::bfloat16);
    if (param.dst_precision == dnnl_s8)
        _output_type_size = sizeof(int8_t);

    _use_position2d = param.use_position2d;
    if (_use_position2d) {
        _rotary_ndims = static_cast<int>(_size_per_head / 2);
    } else {
        _rotary_ndims = static_cast<int>(_size_per_head * _rotary_pct);
    }
    initRotery(_max_seq_len);

    return true;
}

void emb_gpt_impl_avx512::initRotery(size_t max_seq_len) {
    std::vector<float> inv_freq;
    for (int i = 0; i < _rotary_ndims; i += 2) {
        inv_freq.push_back(1.0f / (powf(_rotary_emb_base, static_cast<float>(i) / _rotary_ndims)));
    }
    std::vector<float> t;
    for (size_t i = 0; i < max_seq_len * 2; i++) {
        t.push_back(static_cast<float>(i));
    }
    auto width = _rotary_ndims / 2 * 2;
    auto height = max_seq_len * 2;
    auto capacity = height * width * sizeof(float);
    _cos_cached = std::shared_ptr<float>(
                        reinterpret_cast<float*>(aligned_alloc(64, capacity)),
                        [](void * p) { ::free(p); });
    _sin_cached = std::shared_ptr<float>(
                        reinterpret_cast<float*>(aligned_alloc(64, capacity)),
                        [](void * p) { ::free(p); });

    auto* cos_p = _cos_cached.get();
    auto* sin_p = _sin_cached.get();
    for (size_t i = 0; i < height; i++) {
        for (int j = 0; j < width / 2; j++) {
            cos_p[i * width + j] = cosf(t[i] * inv_freq[j]);
            cos_p[i * width + j + width / 2] = cosf(t[i] * inv_freq[j]);
            sin_p[i * width + j] = sinf(t[i] * inv_freq[j]);
            sin_p[i * width + j + width / 2] = sinf(t[i] * inv_freq[j]);
        }
    }
}

// q_src shape: [batch, q_seq_len, num_attention_heads, 3 * head_size]
// q_dst shape: [batch, num_attention_heads, q_seq_len, head_size_aligned]
// kv_src shape: [batch, q_seq_len, num_attention_heads, 3 * head_size]
// kv_dst shape: [batch, num_attention_heads, q_seq_len+past_seq_len, head_size_aligned]
void emb_gpt_impl_avx512::applyRotaryPosEmbMemcpy(uint8_t* q_src, uint8_t* k_src, uint8_t* v_src, uint8_t* q_dst, uint8_t** k_dst, uint8_t** v_dst,
    size_t batch, size_t q_seq_len, size_t past_seq_len) {
    auto key_offset = _output_type_size * past_seq_len * _size_per_head_aligned;
    auto* cos_cached = _cos_cached.get() + past_seq_len * _rotary_ndims;
    auto* sin_cached = _sin_cached.get() + past_seq_len * _rotary_ndims;
    parallel_for3d(batch, _head_num, q_seq_len, [&](size_t b, size_t h, size_t s) {
        // q, k rotary encoding
        auto q_dst_batch = q_dst + b * _head_num * q_seq_len * _size_per_head_aligned * _output_type_size;
        auto k_dst_batch = k_dst[b] + key_offset;
        auto v_dst_batch = v_dst[b] + key_offset;
        auto q_src_batch = q_src + b * _hidden_size * 3 * q_seq_len * _input_type_size;
        auto k_src_batch = k_src + b * _hidden_size * 3 * q_seq_len * _input_type_size;
        auto v_src_batch = v_src + b * _hidden_size * 3 * q_seq_len * _input_type_size;
        auto q_dst_seq = q_dst_batch + s * _size_per_head_aligned * _output_type_size;
        auto k_dst_seq = k_dst_batch + s * _size_per_head_aligned * _output_type_size;
        auto v_dst_seq = v_dst_batch + s * _size_per_head_aligned * _output_type_size;
        auto q_src_seq = q_src_batch + s * _hidden_size * 3 * _input_type_size;
        auto k_src_seq = k_src_batch + s * _hidden_size * 3 * _input_type_size;
        auto v_src_seq = v_src_batch + s * _hidden_size * 3 * _input_type_size;
        auto* q_src_f = reinterpret_cast<ov::bfloat16*>(q_src_seq + h * _size_per_head * 3 * _input_type_size);
        auto* k_src_f = reinterpret_cast<ov::bfloat16*>(k_src_seq + h * _size_per_head * 3 * _input_type_size);
        auto* q_dst_f = reinterpret_cast<ov::bfloat16*>(q_dst_seq + h * q_seq_len * _size_per_head_aligned * _output_type_size);
        auto* k_dst_f = reinterpret_cast<ov::bfloat16*>(k_dst_seq + h * _max_seq_len * _size_per_head_aligned * _output_type_size);
        rotary_avx512(_rotary_ndims, cos_cached + s * _rotary_ndims, sin_cached + s * _rotary_ndims, q_src_f, k_src_f, q_dst_f, k_dst_f);

        // q, k concat
        memcpy(reinterpret_cast<uint8_t*>(q_dst_f) + _rotary_ndims * _output_type_size, reinterpret_cast<uint8_t*>(q_src_f) + _rotary_ndims * _input_type_size, _output_type_size * (_size_per_head - _rotary_ndims));
        memcpy(reinterpret_cast<uint8_t*>(k_dst_f) + _rotary_ndims * _output_type_size, reinterpret_cast<uint8_t*>(k_src_f) + _rotary_ndims * _input_type_size, _output_type_size * (_size_per_head - _rotary_ndims));
        // v concat
        memcpy(static_cast<uint8_t*>(v_dst_seq) + h * _max_seq_len * _size_per_head_aligned * _output_type_size,
            static_cast<uint8_t*>(v_src_seq) + h * _size_per_head * 3 * _input_type_size,
            _size_per_head * _output_type_size);
    });
}

// q_src shape: [batch, q_seq_len, num_attention_heads, 3 * head_size]
// q_dst shape: [batch, num_attention_heads, q_seq_len, head_size_aligned]
// kv_src shape: [batch, q_seq_len, num_attention_heads, 3 * head_size]
// kv_dst shape: [batch, num_attention_heads, q_seq_len+past_seq_len, head_size_aligned]
// position2d_ids: [batch, 2, q_seq_len]
void emb_gpt_impl_avx512::applyRotaryPosEmbMemcpyWithPosition2d(uint8_t* q_src, uint8_t* k_src, uint8_t* v_src, uint8_t* q_dst, uint8_t** k_dst, uint8_t** v_dst,
    size_t batch, size_t q_seq_len, size_t past_seq_len, int* position2d_ids) {
    auto key_offset = _output_type_size * past_seq_len * _size_per_head_aligned;
    auto* cos_cached = _cos_cached.get();
    auto* sin_cached = _sin_cached.get();
    parallel_for3d(batch, _head_num, q_seq_len, [&](size_t b, size_t h, size_t s) {
        // q, k rotary encoding
        auto q_dst_batch = q_dst + b * _head_num * q_seq_len * _size_per_head_aligned * _output_type_size;
        auto k_dst_batch = k_dst[b] + key_offset;
        auto v_dst_batch = v_dst[b] + key_offset;
        auto pos_batch = position2d_ids + b * 2 * q_seq_len;
        auto block_batch = pos_batch + q_seq_len;
        auto q_src_batch = q_src + b * _hidden_size * 3 * q_seq_len * _input_type_size;
        auto k_src_batch = k_src + b * _hidden_size * 3 * q_seq_len * _input_type_size;
        auto v_src_batch = v_src + b * _hidden_size * 3 * q_seq_len * _input_type_size;
        auto q_dst_seq = q_dst_batch + s * _size_per_head_aligned * _output_type_size;
        auto k_dst_seq = k_dst_batch + s * _size_per_head_aligned * _output_type_size;
        auto v_dst_seq = v_dst_batch + s * _size_per_head_aligned * _output_type_size;
        auto q_src_seq = q_src_batch + s * _hidden_size * 3 * _input_type_size;
        auto k_src_seq = k_src_batch + s * _hidden_size * 3 * _input_type_size;
        auto v_src_seq = v_src_batch + s * _hidden_size * 3 * _input_type_size;
        auto* q_src_f = reinterpret_cast<ov::bfloat16*>(q_src_seq + h * _size_per_head * 3 * _input_type_size);
        auto* k_src_f = reinterpret_cast<ov::bfloat16*>(k_src_seq + h * _size_per_head * 3 * _input_type_size);
        auto* q_dst_f = reinterpret_cast<ov::bfloat16*>(q_dst_seq + h * q_seq_len * _size_per_head_aligned * _output_type_size);
        auto* k_dst_f = reinterpret_cast<ov::bfloat16*>(k_dst_seq + h * _max_seq_len * _size_per_head_aligned * _output_type_size);
        rotary_avx512(_rotary_ndims, cos_cached + pos_batch[s] * _rotary_ndims, sin_cached + pos_batch[s] * _rotary_ndims, q_src_f, k_src_f, q_dst_f, k_dst_f);
        rotary_avx512(_rotary_ndims, cos_cached + block_batch[s] * _rotary_ndims, sin_cached + block_batch[s] * _rotary_ndims,
            q_src_f + _rotary_ndims,
            k_src_f + _rotary_ndims,
            q_dst_f + _rotary_ndims,
            k_dst_f + _rotary_ndims);

        // v concat
        memcpy(static_cast<uint8_t*>(v_dst_seq) + h * _max_seq_len * _size_per_head_aligned * _output_type_size,
            static_cast<uint8_t*>(v_src_seq) + h * _size_per_head * 3 * _input_type_size,
            _size_per_head * _output_type_size);
    });
}

void emb_gpt_impl_avx512::exec(const emb_gpt::exec_param& param) {
    // [batch, seq_len, (num_heads * 3 * head_size)]
    //   --> [batch, seq_len, num_heads, 3 * head_size]
    auto* qkv = param.qkv;
    auto query = qkv;                                             // qkv[..., : self.head_size].permute(0, 2, 1, 3)
    auto key = qkv + _size_per_head * _input_type_size;           // qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
    auto value = qkv + 2 * _size_per_head * _input_type_size;     // qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)
    auto query_dst = param.query_dst;
    auto key_dst = param.layer_past_key_padded;
    auto value_dst = param.layer_past_value_padded;
    auto batch = param.batch;
    auto query_seq_len = param.query_seq_len;
    auto past_seq_len = param.past_seq_len;
    // transpose + rotary embbeding:
    // transpose: [batch, seq_len, num_attention_heads, 3 * head_size] -->
    //          3 [batch, num_attention_heads, seq_len, head_size]
    // rotary embbeding: part of key will write to past_key, part of query will write to tempory buffer
    if (_create_param.dst_precision == dnnl_s8) {
        // query pass part(temp buffer): query = torch.cat((query, query_pass), dim=-1)
        // key pass part(past_key): key = torch.cat((key, key_pass), dim=-1)
        // value(pastKeys): value = torch.cat((past_value, value), dim=-2)
        // applyRotaryPosEmbMemcpyQuant(query, key, queryTranspose.get(), current_k_bufs, _output_type_size * new_seq_offset * _size_per_head_aligned,
        //     _cos_cached.get(), _sin_cached.get(), batch, seq_len, new_seq_offset, value, current_v_bufs);
        assert(false);
    } else {
        // query pass part(temp buffer): query = torch.cat((query, query_pass), dim=-1)
        // key pass part(past_key): key = torch.cat((key, key_pass), dim=-1)
        // value(pastKeys): value = torch.cat((past_value, value), dim=-2)
        // q_dst shape: [batch, num_attention_heads, q_seq_len, head_size_aligned]
        // kv_dst shape: [batch, num_attention_heads, q_seq_len+past_seq_len, head_size_aligned]
        if (_use_position2d) {
            applyRotaryPosEmbMemcpyWithPosition2d(query, key, value, query_dst, key_dst, value_dst, batch, query_seq_len, past_seq_len, param.position2d_ids);
        } else {
            applyRotaryPosEmbMemcpy(query, key, value, query_dst, key_dst, value_dst, batch, query_seq_len, past_seq_len);
        }
    }
}

std::shared_ptr<emb_gpt::impl> new_impl_avx512() {
    return std::make_shared<emb_gpt_impl_avx512>();
}

}