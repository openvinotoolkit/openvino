// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Generic codec-aware multi-head attention pipeline entry point.

#pragma once

#include <cstddef>

#include "codecs/cache_codec.hpp"
#include "cpu_parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

// Generic codec-aware fused multi-head attention pipeline.
// k_codec / v_codec: identify the encoding scheme of cached K/V.
// k_scale_zp / v_scale_zp: scale/zp tensor (empty when cache is raw).
// key_group_size / value_group_size: u8 group size (ignored when not u8).
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
                  ov::Extensions::Cpu::CacheCodec k_codec,
                  ov::Extensions::Cpu::CacheCodec v_codec,
                  bool auto_causal,
                  const ov::intel_cpu::PlainTensor& sink_input,
                  const ov::intel_cpu::CpuParallelPtr& cpu_parallel,
                  const ov::intel_cpu::PlainTensor& k_scale_zp,
                  size_t key_group_size,
                  const ov::intel_cpu::PlainTensor& v_scale_zp,
                  size_t value_group_size,
                  ov::element::Type q_precision,
                  size_t value_head_dim);

}  // namespace ov::Extensions::Cpu::XARCH
