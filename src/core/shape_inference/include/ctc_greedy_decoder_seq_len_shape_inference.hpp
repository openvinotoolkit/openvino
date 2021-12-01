// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/ctc_greedy_decoder_seq_len.hpp>

namespace ov {
namespace op {
namespace v6 {

template <class T>
void shape_infer(const CTCGreedyDecoderSeqLen* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() >= 2 && output_shapes.size() == 2);
    const auto& logits_shape = input_shapes[0];
    const auto& seq_len_shape = input_shapes[1];
    const bool logits_is_static_rank = logits_shape.rank().is_static();
    const bool seq_len_is_static_rank = seq_len_shape.rank().is_static();

    // check ranks of input tensors
    if (logits_is_static_rank) {
        NODE_VALIDATION_CHECK(op,
                              logits_shape.rank().compatible(3),
                              "The rank of logits tensor must be equal to 3.");
    }
    if (seq_len_is_static_rank) {
        NODE_VALIDATION_CHECK(op,
                              seq_len_shape.rank().compatible(1),
                              "The rank of sequence len tensor must be equal to 1.");
    }

    // validate input shapes and compute output shape
    auto& decoded_shape = output_shapes[0];
    auto& seq_shape = output_shapes[1];
    decoded_shape.resize(2);
    seq_shape.resize(1);

    auto& batch_size = decoded_shape[0];
    auto& time_size = decoded_shape[1];

    if (logits_is_static_rank) {
        if (logits_shape[0].is_static()) {
            batch_size = logits_shape[0];
        }
        if (logits_shape[1].is_static()) {
            time_size = logits_shape[1];
        }
    }
    //Batch can be dynamic, if so use seq_len's batch. If both static, two dims must equal
    if (seq_len_is_static_rank && seq_len_shape[0].is_static()) {
        NODE_VALIDATION_CHECK(op,
                              seq_len_shape[0].compatible(batch_size),
                              "The first dimensions of input tensors must match.");
        batch_size = seq_len_shape[0];
    }

    if (logits_is_static_rank && seq_len_is_static_rank) {
        batch_size = seq_len_shape[0] & logits_shape[0];
    }
    seq_shape[0] = batch_size;
}
}  // namespace v6
}  // namespace op
}  // namespace ov