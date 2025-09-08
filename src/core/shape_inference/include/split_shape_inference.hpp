// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/validation_util.hpp"
#include "openvino/op/split.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

/**
 * \brief Shape inference for Split V1 operator.
 *
 * \note The split operation cause label lost on splitted dimension even if number of splits is one,
 * because in this case split will be removed by transformation (as NOP) and in fact label will be propagated.
 *
 * \tparam T             Type of shape.
 *
 * \param op             Split operator pointer.
 * \param input_shapes   Split input shapes.
 * \param ta             Tensor accessor to constant data.
 */
template <typename T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const Split* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2));

    const auto& data_ps = input_shapes[0];
    const auto& axis_ps = input_shapes[1];

    NODE_VALIDATION_CHECK(op, axis_ps.rank().compatible(0), "'axis' input must be a scalar. Got: ", axis_ps);

    TRShape each_output_shape = data_ps;
    const auto data_rank = data_ps.rank();

    auto axes_values = get_input_const_data_as<TRShape, int64_t>(op, 1, ta);
    const auto& num_splits = op->get_num_splits();
    if (axes_values && data_rank.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              axes_values->size() == 1,
                              "a scalar axis value is expected. Got: ",
                              axes_values->size(),
                              " axes");

        const auto axis = ov::util::try_normalize_axis((*axes_values)[0], data_rank, *op);

        if (data_ps[axis].is_static()) {
            const auto dimension_at_axis = data_ps[axis].get_length();

            NODE_VALIDATION_CHECK(op,
                                  dimension_at_axis % num_splits == 0,
                                  "Dimension of data input shape along 'axis': ",
                                  dimension_at_axis,
                                  " must be evenly divisible by 'num_splits' attribute value: ",
                                  num_splits);

            each_output_shape[axis] = dimension_at_axis / num_splits;
        } else {
            const auto dim_interval_at_axis = data_ps[axis].get_interval();
            NODE_VALIDATION_CHECK(op,
                                  dim_interval_at_axis.get_max_val() >= static_cast<int64_t>(num_splits),
                                  "The interval maximum of the dimension for data "
                                  "input shape along 'axis' must be "
                                  "greater or equal to 'num_splits' attribute. Got: ",
                                  dim_interval_at_axis,
                                  " and ",
                                  num_splits);

            auto dim_interval_at_axis_min =
                static_cast<int64_t>(dim_interval_at_axis.get_min_val() * (1.0f / num_splits));
            auto dim_interval_at_axis_max = dim_interval_at_axis.get_max_val();
            if (dim_interval_at_axis.has_upper_bound()) {
                dim_interval_at_axis_max = static_cast<int64_t>(dim_interval_at_axis_max * (1.0f / num_splits));
            }
            each_output_shape[axis] = Dimension(dim_interval_at_axis_min, dim_interval_at_axis_max);
        }
    } else {
        each_output_shape = ov::PartialShape::dynamic(data_ps.rank());
    }

    return {num_splits, each_output_shape};
}

}  // namespace v1
}  // namespace op
}  // namespace ov
