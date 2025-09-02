// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <fstream>
#include <utility>

#include "openvino/core/shape.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

ov::intel_cpu::PlainTensor xattn_estimate(ov::intel_cpu::PlainTensor& query,
                                          ov::intel_cpu::PlainTensor& key,
                                          size_t block_size,
                                          size_t stride,
                                          int norm,
                                          float threshold,
                                          bool causal);

}  // namespace ov::Extensions::Cpu::XARCH
