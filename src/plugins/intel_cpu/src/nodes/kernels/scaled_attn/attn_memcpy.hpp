// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "cpu_parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

void attn_memcpy(const ov::intel_cpu::PlainTensor& k_input,
                 const ov::intel_cpu::PlainTensor& v_input,
                 const ov::intel_cpu::PlainTensor& past_k_output,
                 const ov::intel_cpu::PlainTensor& past_v_output,
                 const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

void paged_attn_memcpy(const ov::intel_cpu::PlainTensor& k_input,
                       const ov::intel_cpu::PlainTensor& v_input,
                       const ov::intel_cpu::PlainTensor& past_k_output,
                       const ov::intel_cpu::PlainTensor& past_v_output,
                       const ov::intel_cpu::PlainTensor& slot_mapping,
                       const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

// Per-tensor (K or V) copy into cache at position L0, parallelized over B×H.
// Handles same-precision memcpy and f32→f16/bf16 SIMD conversion.
void attn_memcpy2d(const ov::intel_cpu::PlainTensor& src,
                   const ov::intel_cpu::PlainTensor& dst,
                   size_t L0,
                   const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

void attn_memcpy2d_kernel(void* src,
                          void* dst,
                          ov::element::Type src_type,
                          ov::element::Type dst_type,
                          size_t src_stride,
                          size_t dst_stride,
                          size_t width,
                          size_t height);

}  // namespace ov::Extensions::Cpu::XARCH
