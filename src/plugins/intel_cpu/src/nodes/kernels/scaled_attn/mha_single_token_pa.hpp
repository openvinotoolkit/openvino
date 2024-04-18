// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <openvino/core/type/element_type.hpp>
#include "utils/plain_tensor.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void mha_single_token_pa(const ov::intel_cpu::PlainTensor& query,
                         const ov::intel_cpu::PlainTensor& present_key,
                         const ov::intel_cpu::PlainTensor& present_value,
                         const ov::intel_cpu::PlainTensor& block_tables,
                         size_t max_context_len,
                         const ov::intel_cpu::PlainTensor& context_lens,
                         ov::intel_cpu::PlainTensor& output_emb,
                         ov::intel_cpu::PlainTensor& buf_attn_w,
                         ov::intel_cpu::PlainTensor& buf_attn_score,
                         float d_scale);

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov