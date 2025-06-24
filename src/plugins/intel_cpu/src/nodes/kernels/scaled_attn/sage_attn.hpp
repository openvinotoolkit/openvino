// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/rope.h"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "paged_attn_kernel.hpp"
#include "softmax_kernel.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"
#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#if defined(OPENVINO_ARCH_X86_64)
#    include "nodes/kernels/x64/brgemm_kernel.hpp"
#endif

#include <cstddef>
#include <cstdint>

namespace ov::Extensions::Cpu::XARCH {
#if defined(OPENVINO_ARCH_X86_64)
void sage_attn_transpose_k(const ReorderWorkItem& item,
                           const size_t hk,
                           const size_t block_size,
                           const std::shared_ptr<ov::intel_cpu::BrgemmKernel>& brgemm_ptr,
                           ov::intel_cpu::PlainTensor& key_cache,
                           ov::intel_cpu::PlainTensor& qk_scratch_b);

void sage_attn_transpose_k(const ReorderWorkItem& item,
                           const size_t hk,
                           const size_t block_size,
                           const std::shared_ptr<ov::intel_cpu::BrgemmKernel>& brgemm_ptr,
                           ov::intel_cpu::PlainTensor& key_cache,
                           ov::intel_cpu::PlainTensor& qk_scratch_b) {
    const auto batch_in_reorder = item.batch_in_reorder;
    const auto kv_block = item.kv_block_id;
    const auto block_number = item.block_number;
    const auto S = key_cache.m_dims[3] - sizeof(float);
    if (block_number < 0) {
        return;
    }
    const size_t valid_len = item.valid_block_len;
    // indexing as i8
    auto* k_ptr = key_cache.ptr<int8_t, ov::element::i8>(block_number, hk, 0, sizeof(float));
    for (size_t i = valid_len; i < block_size; i++) {
        memset(key_cache.ptr<int8_t, ov::element::i8>(block_number, hk, i), 0, sizeof(int8_t) * key_cache.m_dims[3]);
    }
    auto* repacked = qk_scratch_b.ptr<int8_t>(batch_in_reorder, kv_block, hk);
    brgemm_ptr->copy_buffer_b(k_ptr, repacked);
    // layout of repacked_data
    // block_size * S int8(quantized key)
    // block_size * scales (FP32)
    // copy b_scale to dst tensor
    float* scales = reinterpret_cast<float*>(qk_scratch_b.ptr<int8_t>(batch_in_reorder, kv_block, hk, block_size * S));
    for (size_t i = 0; i < valid_len; i++) {
        scales[i] = reinterpret_cast<float*>(key_cache.ptr<int8_t, ov::element::i8>(block_number, hk, i, 0))[0];
    }
}
#endif
template <typename DATA_TYPE, ov::element::Type_t KEY_PREC>
void sage_attn_quantize_q(const ov::intel_cpu::PlainTensor& q,
                          ov::intel_cpu::PlainTensor& quantized_q,
                          const ov::intel_cpu::PlainTensor& past_lens,
                          const ov::intel_cpu::PlainTensor& subsequence_begins) {
    size_t H = q.m_dims[1];
    size_t S = q.m_dims[3];
    parallel_for2d(past_lens.size(0), H, [&](size_t sub_seq_id, size_t h) {
        const auto q_len =
            subsequence_begins.ptr<int32_t>()[sub_seq_id + 1] - subsequence_begins.ptr<int32_t>()[sub_seq_id];
        const auto batch_in_token = subsequence_begins.ptr<int32_t>()[sub_seq_id];
        if (q_len > 1) {
            parallel_for(q_len, [&](int32_t l) {
                quantize_q_by_dims<DATA_TYPE, KEY_PREC>(q, quantized_q, batch_in_token + l, h, S);
            });
        }
    });
}
}  // namespace ov::Extensions::Cpu::XARCH