// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

void recurrent_linear_attn(const ov::intel_cpu::PlainTensor& query,
                      const ov::intel_cpu::PlainTensor& key,
                      const ov::intel_cpu::PlainTensor& value,
                      const ov::intel_cpu::PlainTensor& beta,
                      const ov::intel_cpu::PlainTensor& g,
                      const ov::intel_cpu::PlainTensor& initial_states,
                      ov::intel_cpu::PlainTensor& output,
                      ov::intel_cpu::PlainTensor& output_hidden_states);

}  // namespace ov::Extensions::Cpu::XARCH
