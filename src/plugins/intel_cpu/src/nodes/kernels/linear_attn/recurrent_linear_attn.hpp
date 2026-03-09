// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "cpu_parallel.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

void recurrent_linear_attn(const ov::intel_cpu::PlainTensor& query,
                           const ov::intel_cpu::PlainTensor& key,
                           const ov::intel_cpu::PlainTensor& value,
                           const ov::intel_cpu::PlainTensor& recurrent_state,
                           const ov::intel_cpu::PlainTensor& gate,
                           const ov::intel_cpu::PlainTensor& beta,
                           float eps,
                           bool fuse_qk_l2norm,
                           bool fuse_q_scale,
                           ov::intel_cpu::PlainTensor& output_attn,
                           ov::intel_cpu::PlainTensor& output_recurrent_state,
                           ov::intel_cpu::PlainTensor& temp_buffer,
                           const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

}  // namespace ov::Extensions::Cpu::XARCH
