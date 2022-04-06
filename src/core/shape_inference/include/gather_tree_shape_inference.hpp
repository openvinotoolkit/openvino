// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/gather_tree.hpp>

namespace ov {
namespace op {
namespace v1 {
template <class T>
void shape_infer(const GatherTree* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4 && output_shapes.size() == 1);
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    const auto& step_ids_pshape = input_shapes[0];
    const auto& parent_idx_pshape = input_shapes[1];
    const auto& max_seq_len_pshape = input_shapes[2];
    const auto& end_token_pshape = input_shapes[3];
    auto& result_pshape = output_shapes[0];
    result_pshape = step_ids_pshape;
    NODE_VALIDATION_CHECK(op,
                          T::merge_into(result_pshape, parent_idx_pshape) && result_pshape.rank().compatible(3),
                          "step_ids and parent_idx inputs must have the same shape with rank 3. Got: ",
                          step_ids_pshape,
                          " and ",
                          parent_idx_pshape,
                          ", respectively");

    NODE_VALIDATION_CHECK(op,
                          max_seq_len_pshape.rank().compatible(1),
                          "max_seq_len input must have rank 1. Got: ",
                          max_seq_len_pshape);

    if (result_pshape.rank().is_static() && max_seq_len_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op,
                              DimType::merge(result_pshape[1], result_pshape[1], max_seq_len_pshape[0]),
                              "Number of elements of max_seq_len input must match BATCH_SIZE dimension of "
                              "step_ids/parent_idx inputs. Got: ",
                              result_pshape[1],
                              " and ",
                              max_seq_len_pshape[0],
                              ", respectively");
    }

    NODE_VALIDATION_CHECK(op,
                          end_token_pshape.rank().compatible(0),
                          "end_token input must be scalar. Got: ",
                          end_token_pshape);
}
}  // namespace v1
}  // namespace op
}  // namespace ov