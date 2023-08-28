// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "ngraph/type/element_type.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

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
