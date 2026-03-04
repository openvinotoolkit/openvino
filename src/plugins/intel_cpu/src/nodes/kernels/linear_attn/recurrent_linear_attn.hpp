// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

void recurrent_linear_attn(const ov::intel_cpu::PlainTensor& query,
                           const ov::intel_cpu::PlainTensor& key,
                           const ov::intel_cpu::PlainTensor& value,
                           const ov::intel_cpu::PlainTensor& recurrent_state,
                           const ov::intel_cpu::PlainTensor& gate,
                           const ov::intel_cpu::PlainTensor& beta,
                           ov::intel_cpu::PlainTensor& output_attn,
                           ov::intel_cpu::PlainTensor& output_recurrent_state,
                           ov::intel_cpu::PlainTensor& temp_buffer);

}  // namespace ov::Extensions::Cpu::XARCH
