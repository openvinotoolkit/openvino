// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfenv>
#include <cmath>
#include <numeric>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
void transpose(const char* data,
               char* out,
               const Shape& data_shape,
               size_t element_size,
               const int64_t* axes_order,
               Shape out_shape);
}  // namespace reference
}  // namespace ov
