// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/gather_tree.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const GatherTree* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4);
    using DimType = typename TShape::value_type;

    const auto& step_ids_shape = input_shapes[0];
    const auto& parent_idx_shape = input_shapes[1];
    const auto& max_seq_len_pshape = input_shapes[2];
    const auto& end_token_pshape = input_shapes[3];

    auto output_shapes = std::vector<TRShape>{step_ids_shape};
    auto& result_shape = output_shapes[0];
    NODE_VALIDATION_CHECK(op,
                          TRShape::merge_into(result_shape, parent_idx_shape) && result_shape.rank().compatible(3),
                          "step_ids and parent_idx inputs must have the same shape with rank 3. Got: ",
                          step_ids_shape,
                          " and ",
                          parent_idx_shape,
                          ", respectively");

    NODE_VALIDATION_CHECK(op,
                          max_seq_len_pshape.rank().compatible(1),
                          "max_seq_len input must have rank 1. Got: ",
                          max_seq_len_pshape);

    if (result_shape.rank().is_static() && max_seq_len_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op,
                              DimType::merge(result_shape[1], result_shape[1], max_seq_len_pshape[0]),
                              "Number of elements of max_seq_len input must match BATCH_SIZE dimension of "
                              "step_ids/parent_idx inputs. Got: ",
                              result_shape[1],
                              " and ",
                              max_seq_len_pshape[0],
                              ", respectively");
    }

    NODE_VALIDATION_CHECK(op,
                          end_token_pshape.rank().compatible(0),
                          "end_token input must be scalar. Got: ",
                          end_token_pshape);
    return output_shapes;
}
}  // namespace v1
}  // namespace op
}  // namespace ov
