// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

void attn_quantkv(const ov::intel_cpu::PlainTensor& k_src,
                  const ov::intel_cpu::PlainTensor& v_src,
                  float* temp_buffer,
                  const ov::intel_cpu::PlainTensor& k_dst,
                  const ov::intel_cpu::PlainTensor& v_dst,
                  const ov::intel_cpu::PlainTensor& k_scale_zp,
                  const ov::intel_cpu::PlainTensor& v_scale_zp,
                  const size_t L0,
                  const bool quant_k_by_channel,
                  const size_t k_group_size,
                  const size_t v_group_size);

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
                        const bool quant_key_by_channel,
                        const size_t key_group_size,
                        const size_t value_group_size);

void attn_quant_u8(const float* src, uint8_t* dst, size_t n, float& scale, float& zp);

void attn_dequant_u8(const uint8_t* src, float* dst, size_t n, float scale, float zp);

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
