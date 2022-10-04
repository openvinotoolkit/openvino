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
        const auto out_rank_size = input_shape.rank().get_length();

        if (axes.empty()) {
            generate_transpose_default_order(axes, out_rank_size);
        } else {
            NODE_VALIDATION_CHECK(op,
                                  is_valid_axes_order(axes, out_rank_size),
                                  "Permutation ",
                                  AxisVector(axes.begin(), axes.end()),
                                  " is not valid for input shape ",
                                  input_shape);
        }

        output_shape.resize(out_rank_size);
        std::transform(axes.cbegin(), axes.cend(), output_shape.begin(), [&input_shape](const size_t axis) {
            return input_shape[axis];
        });
    } else {
        output_shape =
            has_order ? ov::PartialShape::dynamic(axes.size()) : ov::PartialShape::dynamic(input_shape.rank());
    }
}
}  // namespace v1
}  // namespace op
}  // namespace ov
