// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/model.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_ti_with_lstm_cell(ov::element::Type type = ov::element::f32,
                                                  size_t N = 32,   // Batch size
                                                  size_t L = 10,   // Sequence length
                                                  size_t I = 8,    // Input size
                                                  size_t H = 32);  // Hidden size
}  // namespace utils
}  // namespace test
}  // namespace ov