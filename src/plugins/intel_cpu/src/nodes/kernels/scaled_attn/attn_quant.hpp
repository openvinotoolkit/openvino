// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>

#include "cpu_parallel.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu {
struct QuantizeParam {
    bool quant_key_by_channel;
    bool quant_value_by_channel;
    bool is_sage_attn;
    size_t key_group_size;
    size_t value_group_size;
};
}  // namespace ov::Extensions::Cpu

namespace ov::Extensions::Cpu::XARCH {

void paged_attn_quantkv(const ov::intel_cpu::PlainTensor& k_src,
                        const ov::intel_cpu::PlainTensor& v_src,
                        const ov::intel_cpu::PlainTensor& k_dst,
                        const ov::intel_cpu::PlainTensor& v_dst,
                        const ov::intel_cpu::PlainTensor& past_lens,
                        const ov::intel_cpu::PlainTensor& subsequence_begins,
                        const ov::intel_cpu::PlainTensor& block_indices,
                        const ov::intel_cpu::PlainTensor& block_indices_begins,
                        const ov::intel_cpu::PlainTensor& slot_mapping,
                        ov::intel_cpu::PlainTensor& temp_buffer,
                        const QuantizeParam& quant_param,
                        const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

void attn_quant_u8(const float* src, uint8_t* dst, size_t n, float& scale, float& zp);

void attn_dequant_u8(const uint8_t* src, float* dst, size_t n, float* params);

// Per-tensor (K or V) u8 quantize into cache with per-group scale/zp.
// Token-level loop replacement for the K/V branch of the removed batched
// attn_quantkv.
void attn_quant_by_token(const ov::intel_cpu::PlainTensor& cur,
                         const ov::intel_cpu::PlainTensor& dst,
                         const ov::intel_cpu::PlainTensor& scale_zp,
                         size_t L0,
                         size_t group_size,
                         const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

// Per-tensor (K or V) by-channel u8 quantization for the concat-SDPA
// compress_cache path. Mirrors the K-side of the removed batched attn_quantkv:
// L0==0 performs fresh per-group quantize; L0>0 dequants the partial leading
// group, appends new tokens, and requantizes.
void attn_quant_by_channel(const ov::intel_cpu::PlainTensor& src,
                           const ov::intel_cpu::PlainTensor& dst,
                           const ov::intel_cpu::PlainTensor& scale_zp,
                           size_t L0,
                           size_t group_size,
                           const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

void attn_quant_by_channel_u8(const float* src,
                              uint8_t* dst,
                              size_t seq_dim,
                              size_t hidden_dims,
                              size_t src_stride,
                              size_t dst_stride,
                              float* scale,
                              float* zp);

void attn_dequant_by_channel_u8(const uint8_t* src,
                                float* dst,
                                size_t seq_dim,
                                size_t hidden_dims,
                                size_t src_stride,
                                size_t dst_stride,
                                float* scale,
                                float* zp);

}  // namespace ov::Extensions::Cpu::XARCH
