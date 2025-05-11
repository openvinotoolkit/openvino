// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "openvino/op/ctc_greedy_decoder_seq_len.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v6 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const CTCGreedyDecoderSeqLen* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 || input_shapes.size() == 3);
    using DimType = typename TShape::value_type;

    const auto& logits_shape = input_shapes[0];
    const auto& seq_len_shape = input_shapes[1];

    if (input_shapes.size() == 3 && input_shapes[2].is_static()) {
        const auto& blank_shape = input_shapes[2];
        const auto blank_is_scalar = blank_shape.size() == 0;
        const auto blank_has_one_elem = blank_shape.size() == 1 && blank_shape[0].get_length() == 1;
        NODE_VALIDATION_CHECK(op,
                              blank_is_scalar || blank_has_one_elem,
                              "Expected 0D or 1D tensor for the 'blank_index' input. Got: ",
                              blank_shape);
    }

    DimType batch_size{};
    DimType time_size{};

    if (logits_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op, logits_shape.size() == 3, "The rank of logits tensor must be equal to 3.");
        batch_size = logits_shape[0];
        time_size = logits_shape[1];
    }
    if (seq_len_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op, seq_len_shape.size() == 1, "The rank of sequence len tensor must be equal to 1.");
        NODE_VALIDATION_CHECK(op,
                              DimType::merge(batch_size, batch_size, seq_len_shape[0]),
                              "The first dimensions of input tensors must match.");
    }

    return {TRShape{batch_size, std::move(time_size)}, TRShape{batch_size}};
}
}  // namespace v6
}  // namespace op
}  // namespace ov
