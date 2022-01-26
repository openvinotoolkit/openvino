// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/ctc_greedy_decoder_seq_len.hpp>

namespace ov {
namespace op {
namespace v6 {

template <class T>
void shape_infer(const CTCGreedyDecoderSeqLen* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2 || input_shapes.size() == 3) && output_shapes.size() == 2);
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    const auto& logits_shape = input_shapes[0];
    const auto& seq_len_shape = input_shapes[1];
    const bool logits_is_static_rank = logits_shape.rank().is_static();
    const bool seq_len_is_static_rank = seq_len_shape.rank().is_static();
    auto& decoded_shape = output_shapes[0];
    auto& seq_shape = output_shapes[1];
    decoded_shape.resize(2);
    seq_shape.resize(1);
    if (input_shapes.size() == 3) {
        const auto& blank_shape = input_shapes[2];
        const auto& blank_rank = blank_shape.rank();
        if (blank_shape.is_static()) {
            const auto blank_is_scalar = blank_rank.get_length() == 0;
            const auto blank_has_one_elem = blank_rank.get_length() == 1 && blank_shape[0].get_length() == 1;
            NODE_VALIDATION_CHECK(op,
                                  blank_is_scalar || blank_has_one_elem,
                                  "Expected 0D or 1D tensor for the 'blank_index' input. Got: ",
                                  blank_shape);
        }
    }
    auto& batch_size = decoded_shape[0];
    auto& time_size = decoded_shape[1];

    // check ranks of input tensors
    if (logits_is_static_rank) {
        NODE_VALIDATION_CHECK(op, logits_shape.rank().compatible(3), "The rank of logits tensor must be equal to 3.");
        batch_size = logits_shape[0];
        time_size = logits_shape[1];
    }
    if (seq_len_is_static_rank) {
        NODE_VALIDATION_CHECK(op,
                              seq_len_shape.rank().compatible(1),
                              "The rank of sequence len tensor must be equal to 1.");
        NODE_VALIDATION_CHECK(op,
                              DimType::merge(batch_size, batch_size, seq_len_shape[0]),
                              "The first dimensions of input tensors must match.");
    }

    seq_shape[0] = batch_size;
}
}  // namespace v6
}  // namespace op
}  // namespace ov