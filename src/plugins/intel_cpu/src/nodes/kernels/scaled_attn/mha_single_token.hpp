// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

// if there is on sink input, please use a default PlainTensor as sink_input which don't contain any data,
// the softmax operation will use the default formula:
// a[i] = exp(a[i] - max(a));
// result[i] = a[i] / sum(a);
// if the sink_input contain data,
// the softmax formula become:
// a[i] = exp(a[i] - max(a, sink));
// result[i] = a[i] / sum(a, sink);
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
                      bool quant_key_by_channel,
                      const ov::intel_cpu::PlainTensor& sink_input);

}  // namespace ov::Extensions::Cpu::XARCH
