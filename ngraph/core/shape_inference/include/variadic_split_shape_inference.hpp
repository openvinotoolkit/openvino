// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/variadic_split.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

// specilization on dynamic shape
template <typename T>
void shape_infer(const VariadicSplit* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;

    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 3));

    output_shapes.clear();
    auto split_lengths_pshape = input_shapes[2];

    if (split_lengths_pshape.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              split_lengths_pshape.rank().get_length() == 1,
                              "Split lengths should be a 1-D tensor. Got ",
                              split_lengths_pshape.rank(),
                              " instead.");

        const auto num_outputs = split_lengths_pshape[0].get_length();
        const auto data_shape = input_shapes[0];

        std::vector<int64_t> axis_values;
        std::vector<int64_t> split_lengths;
        if (data_shape.rank().is_static() && get_data_as_int64<T>(1, op, axis_values, constant_data) &&
            get_data_as_int64<T>(2, op, split_lengths, constant_data)) {
            NODE_VALIDATION_CHECK(op,
                                  axis_values.size() == 1,
                                  "a scalar axis value is expected. Got: ",
                                  axis_values.size(),
                                  " axes");
            const auto axis_val = axis_values[0];
            // Adjust split axis in case of negatives
            const int64_t axis = ov::normalize_axis(op, axis_val, data_shape.rank());

            // Adjust split lengths in case of negatives
            int64_t sum_of_splits = 0;
            int64_t negative_one = -1;
            for (size_t i = 0; i < split_lengths.size(); i++) {
                NODE_VALIDATION_CHECK(op,
                                      split_lengths[i] >= -1,
                                      "Invalid value ",
                                      split_lengths[i],
                                      " in split lengths input. Should be >= -1.");

                if (split_lengths[i] == -1) {
                    NODE_VALIDATION_CHECK(op,
                                          negative_one == -1,
                                          "Cannot infer split with multiple -1 values at ",
                                          negative_one,
                                          " and ",
                                          i);
                    negative_one = i;
                } else {
                    sum_of_splits += split_lengths[i];
                }
            }
            const auto dimension_at_axis = data_shape[axis];

            if (negative_one >= 0 && dimension_at_axis.is_static()) {
                split_lengths[negative_one] = dimension_at_axis.get_length() - sum_of_splits;
                sum_of_splits += split_lengths[negative_one];
            }
            if (data_shape[axis].is_static()) {
                NODE_VALIDATION_CHECK(op,
                                      sum_of_splits == data_shape[axis].get_length(),
                                      "Total length of splits: ",
                                      sum_of_splits,
                                      " must match the length of the chosen axis: ",
                                      data_shape[axis]);
            }

            for (int64_t output{0}; output < num_outputs; ++output) {
                if (split_lengths.at(output) == -1) {
                    auto tmp_shape = data_shape;
                    tmp_shape[axis] = Dimension::dynamic();
                    output_shapes.push_back(tmp_shape);
                } else {
                    auto tmp_shape = data_shape;
                    tmp_shape[axis] = split_lengths.at(output);
                    output_shapes.push_back(tmp_shape);
                }
            }
        } else {
            for (int64_t output{0}; output < num_outputs; ++output)
                output_shapes.push_back(ov::PartialShape::dynamic());
        }
    }
}

}  // namespace v1
}  // namespace op
}  // namespace ov
