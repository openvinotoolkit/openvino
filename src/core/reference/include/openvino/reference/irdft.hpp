// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
void irdft(const std::vector<float>& input_data,
           const Shape& input_data_shape,
           const std::vector<int64_t>& axes_data,
           float* irdft_result,
           const Shape& fft_output_shape,
           const Shape& irdft_output_shape,
           const int64_t last_signal_size);
}  // namespace reference
}  // namespace ov
