// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <openvino/core/type/element_type.hpp>
#include <vector>

#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

void mha_single_token(const ov::intel_cpu::PlainTensor& query,
                      const ov::intel_cpu::PlainTensor& present_key,
                      const ov::intel_cpu::PlainTensor& present_value,
                      const ov::intel_cpu::PlainTensor& alibi_mask,
                      const ov::intel_cpu::PlainTensor& attention_mask,
                      const ov::intel_cpu::PlainTensor& beams,
                      ov::intel_cpu::PlainTensor& output_emb,
                      ov::intel_cpu::PlainTensor& buf_attn_w,
                      ov::intel_cpu::PlainTensor& buf_attn_score,
                      bool has_out_transpose,
                      bool auto_causal,
                      float d_scale,
                      const ov::intel_cpu::PlainTensor& past_k_scale_zp,
                      const ov::intel_cpu::PlainTensor& past_v_scale_zp,
                      ov::intel_cpu::PlainTensor& head_sum,
                      size_t key_group_size,
                      size_t value_group_size,
                      bool quant_key_by_channel);

}  // namespace ov::Extensions::Cpu::XARCH
