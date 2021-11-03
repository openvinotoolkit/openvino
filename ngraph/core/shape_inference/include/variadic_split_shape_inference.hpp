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
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
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

#if 0
void ngraph::op::v1::VariadicSplit::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v1_VariadicSplit_validate_and_infer_types);
    set_input_is_relevant_to_value(0);
    set_input_is_relevant_to_value(1);
    set_input_is_relevant_to_value(2);

    auto split_lengths_pshape = get_input_partial_shape(2);

    if (split_lengths_pshape.is_static()) {
        NODE_VALIDATION_CHECK(this,
                              split_lengths_pshape.rank().get_length() == 1,
                              "Split lengths should be a 1-D tensor. Got ",
                              split_lengths_pshape.rank(),
                              " instead.");

        const auto num_outputs = split_lengths_pshape[0].get_length();
        const auto data = input_value(0);
        const auto axis_source = input_value(1);
        const auto split_lengths_source = input_value(2);
        const auto data_shape = data.get_partial_shape();
        const auto& data_type = data.get_element_type();

        set_output_size(num_outputs);
        const auto& axis_input_constant = get_constant_from_source(axis_source);
        const auto& split_lengths_constant = get_constant_from_source(split_lengths_source);
        if (data_shape.rank().is_static() && axis_input_constant && split_lengths_constant) {
            const auto axis_val = axis_input_constant->cast_vector<int64_t>()[0];
            // Adjust split axis in case of negatives
            const int64_t axis = ngraph::normalize_axis(this, axis_val, data_shape.rank());

            auto split_lengths = split_lengths_constant->cast_vector<int64_t>();
            // Adjust split lengths in case of negatives
            int64_t sum_of_splits = 0;
            int64_t negative_one = -1;
            for (size_t i = 0; i < split_lengths.size(); i++) {
                NODE_VALIDATION_CHECK(this,
                                      split_lengths[i] >= -1,
                                      "Invalid value ",
                                      split_lengths[i],
                                      " in split lengths input. Should be >= -1.");

                if (split_lengths[i] == -1) {
                    NODE_VALIDATION_CHECK(this,
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
            const auto data_shape_dims = vector<Dimension>{data.get_partial_shape()};
            const auto dimension_at_axis = data_shape_dims.at(axis);

            if (negative_one >= 0 && dimension_at_axis.is_static()) {
                split_lengths[negative_one] = dimension_at_axis.get_length() - sum_of_splits;
                sum_of_splits += split_lengths[negative_one];
            }
            if (data_shape[axis].is_static()) {
                NODE_VALIDATION_CHECK(this,
                                      sum_of_splits == data_shape[axis].get_length(),
                                      "Total length of splits: ",
                                      sum_of_splits,
                                      " must match the length of the chosen axis: ",
                                      data_shape[axis]);
            }

            for (int64_t output{0}; output < num_outputs; ++output) {
                const auto output_split_dim =
                    split_lengths.at(output) == -1 ? Dimension::dynamic() : split_lengths.at(output);
                auto tmp_shape = data_shape_dims;
                tmp_shape.at(axis) = output_split_dim;
                set_output_type(output, data_type, ov::PartialShape{tmp_shape});
            }
        } else {
            for (int64_t output{0}; output < num_outputs; ++output) {
                set_output_type(output, data_type, ov::PartialShape::dynamic());
            }
        }
    }
}

void op::v1::Split::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v1_Split_validate_and_infer_types);
    const element::Type& axis_et = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this,
                          axis_et.is_integral_number(),
                          "Element type of 'axis' input must be integer. Got: ",
                          axis_et);

    NODE_VALIDATION_CHECK(this,
                          m_num_splits > 0,
                          "Attribute 'num_splits' must be greater than zero. Got: ",
                          m_num_splits);

    std::vector<ov::PartialShape>& input_shapes = {get_input_partial_shape(0), get_input_partial_shape(1)};
    std::vector<ov::PartialShape>& output_shapes;
    shape_infer(this, input_shapes, output_shapes);
    for (size_t i = 0; i < m_num_splits; ++i) {
        set_output_type(i, get_input_element_type(0), output_shapes[i]);
    }

    set_input_is_relevant_to_shape(0);
}
#endif