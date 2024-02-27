// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace reference {
void concat(const std::vector<const void*>& args,
            void* out,
            const std::vector<Shape>& in_shapes,
            const Shape& out_shape,
            int64_t concatenation_axis,
            const ov::element::Type& elem_type);

}  // namespace reference
}  // namespace ov
