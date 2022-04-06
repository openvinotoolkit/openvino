// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/ctc_greedy_decoder.hpp>

namespace ov {
namespace op {
namespace v0 {

template <class T>
void shape_infer(const CTCGreedyDecoder* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    // output dynamic rank tensor if all inputs are of dynamic rank
    const auto& logits_pshape = input_shapes[0];
    const auto& seq_mask_pshape = input_shapes[1];
    auto& output_shape = output_shapes[0];
    output_shape.resize(4);
    output_shape[2] = 1;
    output_shape[3] = 1;

    if (logits_pshape.rank().is_dynamic() && seq_mask_pshape.rank().is_dynamic()) {
        return;
    }

    // validate input shapes and compute output shape
    auto& batch_size = output_shape[0];
    auto& time_size = output_shape[1];
    // check ranks of input tensors
    if (logits_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op, logits_pshape.rank().compatible(3), "The rank of logits tensor must be equal to 3.");
        time_size = logits_pshape[0];
        batch_size = logits_pshape[1];
    }

    if (seq_mask_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op,
                              seq_mask_pshape.rank().compatible(2),
                              "The rank of sequence mask tensor must be equal to 2.");
        NODE_VALIDATION_CHECK(op,
                              DimType::merge(time_size, time_size, seq_mask_pshape[0]),
                              "The first dimensions of input tensors must match.");
        NODE_VALIDATION_CHECK(op,
                              DimType::merge(batch_size, batch_size, seq_mask_pshape[1]),
                              "The second dimensions of input tensors must match.");
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov
