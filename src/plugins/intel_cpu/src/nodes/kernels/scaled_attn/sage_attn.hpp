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

#if defined(OPENVINO_ARCH_X86_64)
#    include "nodes/kernels/x64/brgemm_kernel.hpp"
#elif defined(OPENVINO_ARCH_ARM64)
#    include "nodes/kernels/aarch64/brgemm_kernel.hpp"
#endif

#include <cstddef>
#include <cstdint>

namespace ov::Extensions::Cpu::XARCH {
#if defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_ARM64)
inline void sage_attn_transpose_k(const ReorderWorkItem& item,
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

// src shape [h, q_len, S]
// dst shape [block_size, S + sizeof(float)]
template <typename DATA_TYPE, ov::element::Type_t DST_PREC>
void sage_attn_quantize_q(const ov::intel_cpu::PlainTensor& src,
                          ov::intel_cpu::PlainTensor& dst,
                          size_t q_start,
                          size_t q_len,
                          size_t h) {
    size_t S = src.m_dims[2];
    size_t group_size = S;
    constexpr size_t sub_byte_multiplier = DST_PREC == ov::element::u4 ? 2 : 1;
    constexpr size_t param_size = sizeof(float) * (DST_PREC == ov::element::i8 ? 1 : 2);
    for (size_t l = 0; l < q_len; l++) {
        for (size_t src_offset = 0, dst_offset = 0; src_offset < S;
             src_offset += group_size, dst_offset += group_size / sub_byte_multiplier + param_size) {
            auto base = dst.ptr<uint8_t, DST_PREC>(l);
            base += dst_offset;
            auto p = reinterpret_cast<float*>(base);
            uint8_t* ptr = base + param_size;
            quantize<DATA_TYPE, DST_PREC>(src.ptr<DATA_TYPE>(h, q_start + l, src_offset), ptr, group_size, p);
        }
    }
}

}  // namespace ov::Extensions::Cpu::XARCH