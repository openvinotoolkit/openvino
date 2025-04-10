// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
void tile(const char* arg,
          char* out,
          const Shape& in_shape,
          const Shape& out_shape,
          const size_t elem_size,
          const std::vector<int64_t>& repeats);
}
}  // namespace ov
