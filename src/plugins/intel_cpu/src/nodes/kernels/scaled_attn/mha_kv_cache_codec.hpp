// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Generic codec-aware multi-head attention pipeline entry point.

#pragma once

#include <cstddef>

#include "cache_spec.hpp"
#include "cpu_parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

// Packed byte size helper (used by scaled_attn.cpp to size cache buffers).
size_t turboq_head_bytes(int head_dim, int bits);

// Generic codec-aware fused multi-head attention pipeline.
// k_spec / v_spec: per-side cache specification.
// k_scale_zp / v_scale_zp: scale/zp tensor (empty when cache is raw).
// q_precision: element type of q_input (f32, bf16, or f16).
void mha_kv_cache(ov::intel_cpu::PlainTensor& q_input,
                  const ov::intel_cpu::PlainTensor& key_cache,
                  const ov::intel_cpu::PlainTensor& packed_value,
                  const ov::intel_cpu::PlainTensor& alibi_mask,
                  const ov::intel_cpu::PlainTensor& attention_mask,
                  const ov::intel_cpu::PlainTensor& beams,
                  ov::intel_cpu::PlainTensor& output_emb,
                  ov::intel_cpu::PlainTensor& buf_attn_w,
                  ov::intel_cpu::PlainTensor& buf_attn_score,
                  bool has_out_transpose,
                  float d_scale,
                  const ov::Extensions::Cpu::CacheSpec& k_spec,
                  const ov::Extensions::Cpu::CacheSpec& v_spec,
                  bool auto_causal,
                  const ov::intel_cpu::PlainTensor& sink_input,
                  const ov::intel_cpu::CpuParallelPtr& cpu_parallel,
                  const ov::intel_cpu::PlainTensor& k_scale_zp,
                  const ov::intel_cpu::PlainTensor& v_scale_zp,
                  ov::element::Type q_precision,
                  size_t value_head_dim,
                  float* per_thread_head_scratch,
                  size_t per_thread_head_stride,
                  const ov::intel_cpu::PlainTensor& k_quant_meta_data,
                  const ov::intel_cpu::PlainTensor& v_quant_meta_data,
                  const ov::intel_cpu::PlainTensor& wht_signs);

}  // namespace ov::Extensions::Cpu::XARCH
