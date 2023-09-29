// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "openvino/op/multinomial.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v13 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Multinomial* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    using TDim = typename TShape::value_type;

    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);

    const auto& input_shape = input_shapes[0];
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           input_shape.rank().compatible(1) || input_shape.rank().compatible(2),
                           "The rank of the 'input' tensor defining output shape must be either 1 or 2.");

    const auto& num_samples_shape = input_shapes[1];
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           num_samples_shape.compatible(TRShape{}) || num_samples_shape.compatible(TRShape{1}),
                           "Number of samples must be a scalar or one element 1D tensor.");

    auto output_shapes = std::vector<TRShape>(1);
    auto& result_shape = output_shapes[0];
    const auto input_shape_static = input_shape.is_static();
    const auto& num_samples = get_input_const_data_as_shape<TRShape>(op, 1, ta);
    if (num_samples && input_shape_static) {
        auto num_samples_shape = *num_samples;
        NODE_VALIDATION_CHECK(op,
                              num_samples_shape[0].get_min_length() >= 0,
                              "Number of samples must be non-negative. Got number of samples: ",
                              num_samples_shape[0].get_min_length());
        if (input_shape.rank().compatible(2)) {
            num_samples_shape.insert(num_samples_shape.begin(), input_shape[0]);
        }
        result_shape = num_samples_shape;
    } else if (input_shape_static) {
        result_shape.push_back(input_shape[0]);
        result_shape.push_back(TDim::dynamic());
    } else {
        result_shape = ov::PartialShape::dynamic();
    }

    return output_shapes;
}
}  // namespace v13
}  // namespace op
}  // namespace ov
