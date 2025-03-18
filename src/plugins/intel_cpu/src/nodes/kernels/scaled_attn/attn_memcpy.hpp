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

void attn_memcpy(const ov::intel_cpu::PlainTensor& k_input,
                 const ov::intel_cpu::PlainTensor& v_input,
                 const ov::intel_cpu::PlainTensor& past_k_output,
                 const ov::intel_cpu::PlainTensor& past_v_output);

void paged_attn_memcpy(const ov::intel_cpu::PlainTensor& k_input,
                       const ov::intel_cpu::PlainTensor& v_input,
                       const ov::intel_cpu::PlainTensor& past_k_output,
                       const ov::intel_cpu::PlainTensor& past_v_output,
                       const ov::intel_cpu::PlainTensor& slot_mapping);

void attn_memcpy2d_kernel(void* src,
                          void* dst,
                          ov::element::Type src_type,
                          ov::element::Type dst_type,
                          size_t src_stride,
                          size_t dst_stride,
                          size_t width,
                          size_t height);

}  // namespace ov::Extensions::Cpu::XARCH
