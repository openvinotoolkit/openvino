// Copyright (C) 2018-2024 Intel Corporation
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

void attn_quantkv(const ov::intel_cpu::PlainTensor& k_src,
                  const ov::intel_cpu::PlainTensor& v_src,
                  const ov::intel_cpu::PlainTensor& k_dst,
                  const ov::intel_cpu::PlainTensor& v_dst,
                  const ov::intel_cpu::PlainTensor& k_scale_zp,
                  const ov::intel_cpu::PlainTensor& v_scale_zp);

void paged_attn_quantkv(const ov::intel_cpu::PlainTensor& k_src,
                        const ov::intel_cpu::PlainTensor& v_src,
                        const ov::intel_cpu::PlainTensor& k_dst,
                        const ov::intel_cpu::PlainTensor& v_dst,
                        const ov::intel_cpu::PlainTensor& slot_mapping);

void attn_quant_u8(const float* src, uint8_t* dst, size_t n, float& scale, float& zp);

void attn_dequant_u8(const uint8_t* src, float* dst, size_t n, float scale, float zp);

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov