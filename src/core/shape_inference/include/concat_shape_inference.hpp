// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v0 {

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const Concat* op, const std::vector<T>& input_shapes) {
    NODE_VALIDATION_CHECK(op, !input_shapes.empty());
    using DimType = typename T::value_type;

    auto concat_axis = op->get_axis();
    const auto empty_dim = DimType{};

    auto concat_dim = DimType{0};
    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes.front();

    if (std::is_same<T, PartialShape>::value) {
        output_shape = PartialShape::dynamic();
    } else {
        output_shape = input_shapes.front();
        concat_axis = ov::util::try_normalize_axis(op->get_axis(), output_shape.rank(), *op);
        output_shape[concat_axis] = empty_dim;
    }

    for (auto& input : input_shapes) {
        const auto& input_rank = input.rank();
        if (input_rank.is_static()) {
            concat_axis = ov::util::try_normalize_axis(op->get_axis(), input_rank, *op);
            auto in_copy = TRShape(input);
            concat_dim += in_copy[concat_axis];
            in_copy[concat_axis] = empty_dim;

            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   TRShape::merge_into(output_shape, in_copy),
                                   "Argument shapes are inconsistent; they must have the same rank, and must "
                                   "have equal dimension everywhere except on the concatenation axis (axis ",
                                   concat_axis,
                                   ").");
        } else {
            concat_dim += empty_dim;
        }
    }

    if (output_shape.rank().is_static()) {
        output_shape[concat_axis] = std::move(concat_dim);
    }
    return output_shapes;
}
}  // namespace v0
}  // namespace op
}  // namespace ov
