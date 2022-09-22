// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/concat.hpp>

#include "utils.hpp"
namespace ov {
namespace op {
namespace v0 {

template <class T>
void shape_infer(Concat* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() > 0 && output_shapes.size() == 1);
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    auto& output_pshape = output_shapes[0];
    DimType concatenation_axis_output_dim{0};
    output_pshape = input_shapes[0];
    const auto& output_rank = output_pshape.rank();
    if (output_rank.is_static()) {
        if (op->get_concatenation_axis() < 0) {
            op->set_concatenation_axis(op->get_axis() < 0 ? op->get_axis() + output_rank.get_length() : op->get_axis());
        }
    } else {
        output_pshape = ov::PartialShape::dynamic();
        return;
    }

    for (uint64_t i = 0; i < input_shapes.size(); i++) {
        const auto& this_input_shape = input_shapes[i];
        const auto& this_input_rank = this_input_shape.rank();
        if (this_input_rank.is_static()) {

            const auto concat_axis = op->get_concatenation_axis();
            NODE_VALIDATION_CHECK(op,
                                  concat_axis < this_input_rank.get_length() && concat_axis >= 0,
                                  "Concatenation axis (",
                                  concat_axis,
                                  ") is out of bounds [",
                                  -this_input_rank.get_length(),
                                  ", ",
                                  this_input_rank.get_length() - 1,
                                  "] for ",
                                  "argument ",
                                  i,
                                  ", which has shape ",
                                  this_input_shape,
                                  ".");


            for (size_t j = 0; j < output_rank.get_length(); j++) {
                if (j == concat_axis)
                    continue;
                NODE_VALIDATION_CHECK(op,
                        DimType::merge(output_pshape[j], output_pshape[j], this_input_shape[j])
                        && output_rank.compatible(this_input_rank),
                        "Argument shapes are inconsistent; they must have the same rank, and must "
                        "have ",
                        "equal dimension everywhere except on the concatenation axis (axis ",
                        concat_axis,
                        ").");
            }
            concatenation_axis_output_dim += this_input_shape[concat_axis];
        } else {
            concatenation_axis_output_dim += Dimension::dynamic();
        }
    }

    if (output_pshape.rank().is_static()) {
        output_pshape[op->get_concatenation_axis()] = concatenation_axis_output_dim;
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov