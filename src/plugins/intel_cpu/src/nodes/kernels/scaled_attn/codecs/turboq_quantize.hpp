// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// TurboQuant encode: per-head quantize + pack for bits ∈ {3, 4}.
// Outputs: packed indices written to dst; per-head fp32 norm written to a
// separate meta_data tensor (out_norm in the per-head entry point).

#pragma once

#include <cstddef>

#include "codec_kernels.hpp"
#include "cpu_parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

// Quantize one head: norm → sign-flip + normalize → WHT in-place → quant + pack.
// ws: per-thread f32 scratch, ≥ head_dim slots.
// Entry point with external linkage; resolved against cross-compile dispatcher.
void turboq_quantize_head(const void* src,
                          void* dst,
                          float* out_norm,
                          int head_dim,
                          int bits,
                          ov::element::Type src_precision,
                          float* ws,
                          const float* signs);

// Quantize a full cache tensor (K or V) using TurboQuant.
// Writes packed indices to dst and per-head fp32 norm to meta_data[b, h, L0+l, 0].
// ws: per-thread scratch; ws[tid] gives a head_dim-sized f32 buffer.
// Fully-qualified StridedData lives in ov::Extensions::Cpu (outside XARCH) so the
// symbol mangling is identical across per-ISA namespaces in the cross-compile dispatcher.
void turboq_quantize(const ov::intel_cpu::PlainTensor& cur,
                     ov::intel_cpu::PlainTensor& dst,
                     ov::intel_cpu::PlainTensor& meta_data,
                     size_t L0,
                     int bits,
                     const ov::intel_cpu::CpuParallelPtr& cpu_parallel,
                     ov::Extensions::Cpu::StridedData<float> ws,
                     const ov::intel_cpu::PlainTensor& signs);

}  // namespace ov::Extensions::Cpu::XARCH
