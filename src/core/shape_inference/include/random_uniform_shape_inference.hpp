// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "openvino/op/random_uniform.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v8 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const RandomUniform* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3);

    const auto& shape = input_shapes[0];
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           shape.rank().compatible(1),
                           "The rank of the tensor defining output shape must be equal to 1.");

    const auto& min_shape = input_shapes[1];
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           min_shape.compatible(TRShape{}) || min_shape.compatible(TRShape{1}),
                           "Min value must be a scalar or one element 1D tensor.");

    const auto& max_shape = input_shapes[2];
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           max_shape.compatible(TRShape{}) || max_shape.compatible(TRShape{1}),
                           "Max value must be a scalar or one element 1D tensor.");

    if (const auto& const_min = get_input_const_data_as<TRShape, double>(op, 1, ta)) {
        if (const auto& const_max = get_input_const_data_as<TRShape, double>(op, 2, ta)) {
            NODE_VALIDATION_CHECK(op,
                                  const_min->front() < const_max->front(),
                                  "Min value must be less than max value. Got min value: ",
                                  const_min->front(),
                                  ", max value: ",
                                  const_max->front());
        }
    }

    auto output_shapes = std::vector<TRShape>();
    if (auto out_shape = get_input_const_data_as_shape<TRShape>(op, 0, ta)) {
        output_shapes.push_back(std::move(*out_shape));
    } else if (std::is_same<TRShape, PartialShape>::value) {
        output_shapes.push_back(
            PartialShape::dynamic(shape.rank().is_static() ? shape[0].get_max_length() : Rank::dynamic()));
    }

    return output_shapes;
}
}  // namespace v8
}  // namespace op
}  // namespace ov
