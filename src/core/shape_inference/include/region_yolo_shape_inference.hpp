// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iterator>

#include "openvino/core/validation_util.hpp"
#include "openvino/op/region_yolo.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v0 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const RegionYolo* op, const std::vector<TShape>& input_shapes) {
    using TDim = typename TShape::value_type;
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 1));

    const auto& input_shape = input_shapes[0];
    const auto& input_rank = input_shape.rank();
    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           input_rank.compatible(4),
                           "Input must be a tensor of rank 4, but got ",
                           input_rank);

    if (input_rank.is_static()) {
        const auto out_rank = input_shape.size();
        output_shape.reserve(out_rank);

        if (op->get_do_softmax()) {
            const auto axis = ov::util::try_normalize_axis(op->get_axis(), input_rank, *op);
            const auto end_axis = ov::util::try_normalize_axis(op->get_end_axis(), input_rank, *op);

            auto input_it = input_shape.cbegin();
            auto out_it = std::copy_n(input_it, axis + 1, std::back_inserter(output_shape));
            input_it += (axis + 1);

            for (; input_it <= input_shape.cbegin() + end_axis; ++input_it) {
                output_shape[axis] *= *input_it;
            }

            std::copy(input_it, input_shape.end(), out_it);
        } else {
            output_shape = input_shape;
            output_shape[1] = TDim((op->get_num_classes() + op->get_num_coords() + 1) * op->get_mask().size());
        }
    } else {
        output_shape = PartialShape::dynamic(Rank(1, 4));
    }
    return output_shapes;
}
}  // namespace v0
}  // namespace op
}  // namespace ov
