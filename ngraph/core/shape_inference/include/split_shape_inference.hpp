// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/split.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

template <typename T>
void shape_infer(const Split* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2));

    output_shapes.clear();

    const auto& data_ps = input_shapes[0];
    const auto& axis_ps = input_shapes[1];

    NODE_VALIDATION_CHECK(op, axis_ps.rank().compatible(0), "'axis' input must be a scalar. Got: ", axis_ps);

    auto each_output_shape = data_ps;
    const auto data_rank = data_ps.rank();
    const auto axis_input = get_constant_from_source(op->input_value(1));

    NODE_VALIDATION_CHECK(op, axis_ps.rank().compatible(0), "'axis' input must be a scalar. Got: ", axis_ps);

    if (axis_input && data_rank.is_static()) {
        auto axis = axis_input->cast_vector<int64_t>()[0];
        axis = ov::normalize_axis(op, axis, data_rank);

        if (data_ps[axis].is_static()) {
            const auto dimension_at_axis = data_ps[axis].get_length();

            NODE_VALIDATION_CHECK(op,
                                  dimension_at_axis % op->m_num_splits == 0,
                                  "Dimension of data input shape along 'axis': ",
                                  dimension_at_axis,
                                  " must be evenly divisible by 'num_splits' attribute value: ",
                                  op->m_num_splits);

            each_output_shape[axis] = dimension_at_axis / op->m_num_splits;
        } else {
            const auto dim_interval_at_axis = data_ps[axis].get_interval();
            NODE_VALIDATION_CHECK(op,
                                  dim_interval_at_axis.get_max_val() >= static_cast<int64_t>(op->m_num_splits),
                                  "The interval maximum of the dimension for data "
                                  "input shape along 'axis' must be "
                                  "greater or equal to 'num_splits' attribute. Got: ",
                                  dim_interval_at_axis,
                                  " and ",
                                  op->m_num_splits);

            auto dim_interval_at_axis_min =
                static_cast<int64_t>(dim_interval_at_axis.get_min_val() * (1.0f / op->m_num_splits));
            auto dim_interval_at_axis_max = dim_interval_at_axis.get_max_val();
            if (dim_interval_at_axis.has_upper_bound()) {
                dim_interval_at_axis_max = static_cast<int64_t>(dim_interval_at_axis_max * (1.0f / op->m_num_splits));
            }
            each_output_shape[axis] = Dimension(dim_interval_at_axis_min, dim_interval_at_axis_max);
        }
    } else {
        each_output_shape = ov::PartialShape::dynamic(data_ps.rank());
    }

    for (size_t i = 0; i < op->m_num_splits; ++i)
        output_shapes.push_back(each_output_shape);
}

}  // namespace v1
}  // namespace op
}  // namespace ov
