// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "openvino/op/ctc_greedy_decoder.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v0 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const CTCGreedyDecoder* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);
    using DimType = typename TShape::value_type;

    // Output shape rank is always static and equal to 4
    // The last two output shape dimensions are always static and equal to 1
    std::vector<DimType> output_dims(4);
    output_dims[2] = 1;
    output_dims[3] = 1;

    const auto& logits_pshape = input_shapes[0];
    const auto& seq_mask_pshape = input_shapes[1];

    if (logits_pshape.rank().is_dynamic() && seq_mask_pshape.rank().is_dynamic()) {
        return {TShape(std::move(output_dims))};
    }

    auto& batch_size = output_dims[0];
    auto& time_size = output_dims[1];

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
    return {TRShape(std::move(output_dims))};
}
}  // namespace v0
}  // namespace op
}  // namespace ov
