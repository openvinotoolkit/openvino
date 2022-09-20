// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "compare.hpp"
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
 * \param constant_data  Map of constant data.
 */
template <class T>
void shape_infer(const Transpose* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    const auto& input_shape = input_shapes[TransposeIn::ARG];
    auto& output = output_shapes[TransposeOut::ARG_T];

    std::vector<int64_t> axes;

    if (get_data_as_int64<T>(op::TransposeIn::ORDER, op, axes, constant_data) && input_shape.rank().is_static()) {
        const auto out_rank_size = input_shape.rank().get_length();
        const auto max_dim_num = out_rank_size - 1;

        if (axes.empty()) {
            axes.reserve(out_rank_size);
            std::generate_n(std::back_inserter(axes), out_rank_size, SeqGen<size_t, Direction::BACKWARD>(max_dim_num));
        } else {
            NODE_VALIDATION_CHECK(
                op,
                (std::set<size_t>(axes.cbegin(), axes.cend()).size() == out_rank_size &&
                 std::all_of(axes.cbegin(), axes.cend(), cmp::Between<int64_t, cmp::LOWER>(0, out_rank_size))),
                "Permutation ",
                AxisVector(axes.begin(), axes.end()),
                " is not valid for input shape ",
                input_shape);
        }

        output.resize(out_rank_size);
        std::transform(axes.cbegin(), axes.cend(), output.begin(), [&input_shape](const size_t axis) {
            return input_shape[axis];
        });
    } else {
        output = ov::PartialShape::dynamic(input_shape.rank());
    }
}
}  // namespace v1
}  // namespace op
}  // namespace ov
