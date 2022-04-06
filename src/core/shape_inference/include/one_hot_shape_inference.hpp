// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/core/validation_util.hpp>
#include <openvino/op/one_hot.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {
void inline resolve_axis(OneHot* op) {
    const auto& indices_shape = op->get_input_partial_shape(0);
    if (indices_shape.rank().is_static()) {
        const auto indices_rank = indices_shape.rank().get_length();
        op->m_axis = ov::normalize_axis(op, op->m_axis, indices_rank + 1, -indices_rank - 1, indices_rank);
    }
}

template <class T>
void shape_infer(const OneHot* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4 && output_shapes.size() == 1);
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    const auto& indices_shape = input_shapes[0];
    const auto& depth_shape = input_shapes[1];
    const auto& on_value_shape = input_shapes[2];
    const auto& off_value_shape = input_shapes[3];

    NODE_VALIDATION_CHECK(op,
                          depth_shape.is_dynamic() || ngraph::is_scalar(depth_shape.to_shape()),
                          "depth input must be scalar.");

    NODE_VALIDATION_CHECK(op,
                          on_value_shape.is_dynamic() || ngraph::is_scalar(on_value_shape.to_shape()),
                          "on_value input must be scalar.");

    NODE_VALIDATION_CHECK(op,
                          off_value_shape.is_dynamic() || ngraph::is_scalar(off_value_shape.to_shape()),
                          "off_value input must be scalar.");

    auto& result_shape = output_shapes[0];
    std::vector<int64_t> depth_vals;
    bool depth_is_set = get_data_as_int64<T>(1, op, depth_vals, constant_data);
    if (indices_shape.rank().is_static()) {
        // decide result rank
        result_shape = indices_shape;
        const auto indices_rank = indices_shape.rank().get_length();
        const auto axis = ov::normalize_axis(op, op->get_axis(), indices_rank + 1, -indices_rank - 1, indices_rank);

        if (depth_is_set) {
            int64_t depth_val = depth_vals[0];
            NODE_VALIDATION_CHECK(op,
                                  depth_val > 0,
                                  "The value of 'depth' must be a positive number.",
                                  " (got ",
                                  depth_val,
                                  ").");
            result_shape.insert(result_shape.begin() + axis, DimType(depth_val));
        } else {
            result_shape.insert(result_shape.begin() + axis, DimType());
        }
    } else {
        result_shape = PartialShape::dynamic();
    }
}
}  // namespace v1
}  // namespace op
}  // namespace ov
