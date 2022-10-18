// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

/**
 * \brief Calculate transpose output shape.
 *
 * \tparam T           Type of shape
 *
 * \param op           Transpose operator pointer.
 * \param input_shape  Transpose input shape.
 * \param axes_order   Transpose axes order (modified if empty).
 *
 * \return Output shape
 */
template <class T>
T calc_output_shape(const Transpose* const op, const T& input_shape, std::vector<int64_t>& axes_order) {
    const auto output_rank = input_shape.size();

    if (axes_order.empty()) {
        generate_transpose_default_order(axes_order, output_rank);
    } else {
        NODE_VALIDATION_CHECK(op,
                              is_valid_axes_order(axes_order, output_rank),
                              "Permutation ",
                              AxisVector(axes_order.begin(), axes_order.end()),
                              " is not valid for input shape ",
                              input_shape);
    }

    T output_shape;
    for (auto&& axis : axes_order) {
        output_shape.push_back(input_shape[axis]);
    }

    return output_shape;
}

/**
 * \brief Do transpose inference on input and output shapes.
 *
 * \tparam T             Type of inference shapes.
 *
 * \param op             Transpose operator pointer.
 * \param input_shapes   Input shapes of transpose.
 * \param output_shapes  Output shapes of transpose which be modified by inference.
 * \param constant_data  Map of constant data.
 */
template <class T>
void shape_infer(const Transpose* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    const auto& input_shape = input_shapes[Transpose::ARG];
    auto& output_shape = output_shapes[Transpose::ARG_T];

    std::vector<int64_t> axes;
    const auto has_order = get_data_as_int64<T>(Transpose::ORDER, op, axes, constant_data);

    if (has_order && input_shape.rank().is_static()) {
        output_shape = calc_output_shape(op, input_shape, axes);
    } else if (has_order) {
        output_shape = ov::PartialShape::dynamic(axes.size());
    } else {
        output_shape = ov::PartialShape::dynamic(input_shape.rank());
    }
}
}  // namespace v1
}  // namespace op
}  // namespace ov
