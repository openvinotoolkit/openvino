// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/validation_util.hpp>
#include <openvino/op/unique.hpp>
#include <vector>

namespace ov {
namespace op {
namespace v10 {

template <class shape_t>
void shape_infer(const Unique* op,
                 const shape_t& input_shape,
                 std::vector<shape_t>& output_shapes,
                 std::unique_ptr<int64_t> axis = nullptr) {
    int64_t input_tensor_capacity = -1;
    if (input_shape.is_static()) {
        input_tensor_capacity = static_cast<int64_t>(shape_size(input_shape.to_shape()));
    }

    output_shapes[0] = PartialShape::dynamic();
    output_shapes[1] =
        input_tensor_capacity > 0 ? PartialShape{{1, input_tensor_capacity}} : PartialShape{{Dimension::dynamic()}};
    output_shapes[2] = output_shapes[1];
    output_shapes[3] = output_shapes[1];

    if (axis) {
        if (input_shape.rank().is_static()) {
            const auto normalized_axis = ngraph::normalize_axis(op, *axis, input_shape.rank());
            const auto dim_at_axis = input_shape[normalized_axis];
            if (dim_at_axis.is_static()) {
                auto output_shape = input_shape;
                output_shape[normalized_axis] = Dimension{1, dim_at_axis.get_length()};
                output_shapes[0] = output_shape;
            } else {
                auto output_shape = input_shape;
                output_shape[normalized_axis] = Dimension::dynamic();
                output_shapes[0] = output_shape;
            }
        }
    } else {
        // no axis => flattened input tensor
        if (input_shape.is_static()) {
            // between 1 and the total number of input tensor's unique elements
            output_shapes[0] = PartialShape{{Dimension{1, input_tensor_capacity}}};
        } else {
            output_shapes[0] = PartialShape{{Dimension::dynamic()}};
        }
    }
}

}  // namespace v10
}  // namespace op
}  // namespace ov
