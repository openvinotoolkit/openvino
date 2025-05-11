// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
void reorg_yolo(const char* arg, char* out, const Shape& in_shape, int64_t stride, const size_t elem_size);
}
}  // namespace ov
