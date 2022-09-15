// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/transpose.hpp"
#include "openvino/op/util/transpose_attr.hpp"
#include "sequnce_generator.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

/**
 * \brief Do transpose inference on input and output shapes.
 *
 * \tparam T             Type of inference shapes.
 *
 * \param op             Transpose operator pointer.
 * \param input_shapes   Input shapes of transpose.
 * \param output_shapes  Output shapes of transpose which be modified by inference.
 */
template <class T>
void shape_infer(const Transpose* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    const auto& input = input_shapes[TransposeIn::ARG];
    const auto& order = input_shapes[TransposeIn::ORDER];
    auto& output = output_shapes[TransposeOut::ARG_T];

    const auto& out_size = (input.is_dynamic() && order.is_static() && order[0].get_length())
                               ? order[0].get_length()
                               : input.rank().get_length();
    output.resize(out_size);

    if (const auto& order_const = get_constant_from_source(op->input_value(TransposeIn::ORDER))) {
        const auto dims_count = out_size - 1;

        auto axes = order_const->get_axis_vector_val();
        if (axes.empty()) {
            std::generate_n(std::back_inserter(axes), out_size, SeqGen<size_t, Direction::BACKWARD>(dims_count));
        }

        NODE_VALIDATION_CHECK(op,
                              input.rank().is_dynamic() || std::all_of(axes.cbegin(),
                                                                       axes.cend(),
                                                                       [&dims_count](const size_t& axis) {
                                                                           return axis <= dims_count;
                                                                       }),
                              "Permutation ",
                              axes,
                              " is not valid for input shape ",
                              input);
        std::transform(axes.cbegin(), axes.cend(), output.begin(), [&input](const size_t axis) {
            return input[axis];
        });
    }
}
}  // namespace v1
}  // namespace op
}  // namespace ov
