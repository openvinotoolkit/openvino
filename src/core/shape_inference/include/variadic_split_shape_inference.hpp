// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/variadic_split.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

template <typename T>
void shape_infer(const VariadicSplit* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    constexpr bool is_dynamic_shape = std::is_base_of<ov::PartialShape, T>::value;

    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 3));

    output_shapes.clear();
    auto axis_pshape = input_shapes[1];
    auto split_lengths_pshape = input_shapes[2];

    NODE_VALIDATION_CHECK(op,
                          axis_pshape.rank().compatible(0) || axis_pshape.compatible({1}),
                          "Axis should be a scalar or of shape [1]. Got ",
                          axis_pshape,
                          " instead.");

    if (split_lengths_pshape.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              split_lengths_pshape.size() == 1,
                              "Split lengths should be a 1-D tensor. Got ",
                              split_lengths_pshape.size(),
                              " instead.");

        const auto num_outputs = split_lengths_pshape[0].get_length();
        const auto& data_shape = input_shapes[0];

        std::vector<int64_t> axis_values;
        std::vector<int64_t> split_lengths;
        if (data_shape.rank().is_static() && get_data_as_int64<T>(1, op, axis_values, constant_data)) {
            NODE_VALIDATION_CHECK(op,
                                  axis_values.size() == 1,
                                  "a scalar axis value is expected. Got: ",
                                  axis_values.size(),
                                  " axes");
            const auto axis_val = axis_values[0];
            // Adjust split axis in case of negatives
            const int64_t axis = ov::normalize_axis(op, axis_val, data_shape.rank());

            if (get_data_as_int64<T>(2, op, split_lengths, constant_data)) {
                // Adjust split lengths in case of negatives
                int64_t sum_of_splits = 0;
                int64_t negative_one_idx = -1;
                for (size_t i = 0; i < split_lengths.size(); i++) {
                    NODE_VALIDATION_CHECK(op,
                                          split_lengths[i] >= -1,
                                          "Invalid value ",
                                          split_lengths[i],
                                          " in split lengths input. Should be >= -1.");

                    if (split_lengths[i] == -1) {
                        NODE_VALIDATION_CHECK(op,
                                              negative_one_idx == -1,
                                              "Cannot infer split with multiple -1 values at ",
                                              negative_one_idx,
                                              " and ",
                                              i);
                        negative_one_idx = i;
                    } else {
                        sum_of_splits += split_lengths[i];
                    }
                }
                const auto dimension_at_axis = data_shape[axis];

                if (negative_one_idx >= 0 && dimension_at_axis.is_static()) {
                    split_lengths[negative_one_idx] = dimension_at_axis.get_length() - sum_of_splits;
                    sum_of_splits += split_lengths[negative_one_idx];
                }
                if (data_shape[axis].is_static()) {
                    NODE_VALIDATION_CHECK(op,
                                          sum_of_splits == data_shape[axis].get_length(),
                                          "Total length of splits: ",
                                          sum_of_splits,
                                          " must match the length of the chosen axis: ",
                                          data_shape[axis]);
                }

                for (auto output = 0; output < num_outputs; ++output) {
                    if (split_lengths.at(output) == -1) {
                        auto out_shape = data_shape;
                        out_shape[axis] = Dimension::dynamic();
                        output_shapes.push_back(out_shape);
                    } else {
                        auto out_shape = data_shape;
                        out_shape[axis] = split_lengths.at(output);
                        output_shapes.push_back(out_shape);
                    }
                }
            } else {
                // we know num_outputs & axis but split_lengths, pass other dimensions besides axis in dynamic shape
                // case
                NODE_VALIDATION_CHECK(op, is_dynamic_shape, "Cannot infer static shape due to lack of split_lengths.");

                auto out_shape = data_shape;
                out_shape[axis] = Dimension::dynamic();
                output_shapes.resize(num_outputs, out_shape);
            }
        } else {
            // we only know num_outputs, only predict the rank
            auto out_shape = ov::PartialShape::dynamic(data_shape.rank());
            output_shapes.resize(num_outputs, out_shape);
        }
    } else {
        // we don't even known the number of outputs in this case.
        // just leave output_shapes as empty.
    }
}

}  // namespace v1
}  // namespace op
}  // namespace ov
