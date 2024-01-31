// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void attn_quantkv(const ov::intel_cpu::PlainTensor& k_input,
                  const ov::intel_cpu::PlainTensor& v_input,
                  const ov::intel_cpu::PlainTensor& past_k_output,
                  const ov::intel_cpu::PlainTensor& past_v_output,
                  const ov::intel_cpu::PlainTensor& past_k_scale_zp,
                  const ov::intel_cpu::PlainTensor& past_v_scale_zp);

void attn_quant_u8(uint8_t* a, float* b, size_t n, float& scale, float& zp);

void attn_dequant_u8(uint8_t* a, float* b, size_t n, float scale, float zp);

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov