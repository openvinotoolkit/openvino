// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <numeric>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
void shuffle_channels(const char* arg,
                      char* out,
                      const Shape& data_shape,
                      size_t elem_size,
                      const int64_t axis,
                      const int64_t group);
}  // namespace reference
}  // namespace ov
