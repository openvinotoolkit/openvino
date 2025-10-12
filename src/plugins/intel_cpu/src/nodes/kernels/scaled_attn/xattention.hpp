// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <fstream>
#include <utility>

#include "openvino/core/shape.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

ov::intel_cpu::PlainTensor xattn_estimate(const ov::intel_cpu::PlainTensor& query,
                                          const ov::intel_cpu::PlainTensor& key,
                                          size_t block_size,
                                          size_t stride,
                                          float threshold);

}  // namespace ov::Extensions::Cpu::XARCH
