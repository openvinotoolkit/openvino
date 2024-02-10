// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
void concat(const std::vector<const char*>& args,
            char* out,
            const std::vector<Shape>& in_shapes,
            const Shape& out_shape,
            int64_t concatenation_axis,
            size_t elem_size);

void concat(const std::vector<const std::string*>& args,
            std::string* out,
            const std::vector<Shape>& in_shapes,
            const Shape& out_shape,
            int64_t concatenation_axis,
            size_t);

}  // namespace reference
}  // namespace ov
