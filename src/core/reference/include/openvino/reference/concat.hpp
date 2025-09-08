// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace reference {
void concat(const std::vector<const char*>& args,
            char* out,
            const std::vector<Shape>& in_shapes,
            const Shape& out_shape,
            int64_t concatenation_axis,
            size_t elem_size,
            const ov::element::Type& elem_type = ov::element::Type_t::dynamic);

}  // namespace reference
}  // namespace ov
