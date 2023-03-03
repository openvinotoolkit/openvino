// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v0 {

template <class T>
void shape_infer(const Concat* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;

    const auto concat_axis = op->get_concatenation_axis();
    const auto empty_dim = DimType{};

    auto concat_dim = DimType{0};
    auto& output_shape = output_shapes.front();

    if (std::is_same<T, PartialShape>::value) {
        output_shape = PartialShape::dynamic();
    } else {
        output_shape = input_shapes.front();
        output_shape[concat_axis] = empty_dim;
    }

    for (auto input : input_shapes) {
        if (input.rank().is_static()) {
            concat_dim += input[concat_axis];
            input[concat_axis] = empty_dim;

            NODE_VALIDATION_CHECK(op,
                                  T::merge_into(output_shape, input),
                                  "Argument shapes are inconsistent; they must have the same rank, and must "
                                  "have ",
                                  "equal dimension everywhere except on the concatenation axis (axis ",
                                  concat_axis,
                                  ").");
        } else {
            concat_dim += empty_dim;
        }
    }

    if (output_shape.rank().is_static()) {
        output_shape[concat_axis] = concat_dim;
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov
