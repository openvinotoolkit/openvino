// Copyright (C) 2018-2025 Intel Corporation
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
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);

    const auto& input_shape = input_shapes[0];
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           input_shape.rank().compatible(2),
                           "Input probabilities must be a 2D tensor.");

    const auto& num_samples_shape = input_shapes[1];
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           num_samples_shape.compatible(TRShape{}) || num_samples_shape.compatible(TRShape{1}),
                           "Number of samples must be a scalar or one element 1D tensor.");

    auto output_shapes = std::vector<TRShape>(1);
    auto& result_shape = output_shapes[0];
    const auto input_rank_static = input_shape.rank().is_static();
    if (input_rank_static) {
        result_shape.push_back(input_shape[0]);
        const auto& num_samples = get_input_const_data_as_shape<TRShape>(op, 1, ta);
        if (num_samples) {
            NODE_VALIDATION_CHECK(op,
                                  (*num_samples)[0].get_min_length() >= 0,
                                  "Number of samples must be non-negative. Got number of samples: ",
                                  (*num_samples)[0].get_min_length());
            result_shape.push_back((*num_samples)[0]);
        } else {
            result_shape.push_back(ov::Dimension::dynamic());
        }
    } else {
        result_shape = ov::PartialShape::dynamic();
    }

    return output_shapes;
}
}  // namespace v13
}  // namespace op
}  // namespace ov
