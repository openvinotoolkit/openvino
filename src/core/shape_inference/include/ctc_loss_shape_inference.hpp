// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <array>

#include "openvino/op/ctc_loss.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v4 {

namespace ctc_loss {
constexpr auto shape_names =
    std::array<const char*, 5>{"logits", "logit length", "labels", "label length", "blank index"};
constexpr auto shape_ranks = std::array<int64_t, 4>{3, 1, 2, 1};
}  // namespace ctc_loss

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const CTCLoss* op, const std::vector<TShape>& input_shapes) {
    using DimType = typename TShape::value_type;
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4 || input_shapes.size() == 5);

    // check ranks of input tensors
    for (size_t i = 0; i < ctc_loss::shape_ranks.size(); ++i) {
        NODE_VALIDATION_CHECK(op,
                              input_shapes[i].rank().compatible(ctc_loss::shape_ranks[i]),
                              "Expected a ",
                              ctc_loss::shape_ranks[i],
                              "D tensor for ",
                              ctc_loss::shape_names[i],
                              ". Got: ",
                              input_shapes[i]);
    }

    // check optional input shape: blank index
    if (input_shapes.size() == 5) {
        const auto& blank_index_pshape = input_shapes[4];
        NODE_VALIDATION_CHECK(op,
                              blank_index_pshape.rank().compatible(0),
                              "Expected a scalar for blank index. Got: ",
                              blank_index_pshape);
    }

    const auto& logits_pshape = input_shapes[0];
    const auto& logits_rank = logits_pshape.rank();

    const auto& logit_length_pshape = input_shapes[1];
    const auto& labels_pshape = input_shapes[2];
    const auto& label_length_pshape = input_shapes[3];

    // check shapes of input tensors
    DimType batch_size = logits_rank.is_static() ? logits_pshape[0] : -1;
    DimType time_steps = logits_rank.is_static() ? logits_pshape[1] : -1;

    NODE_VALIDATION_CHECK(
        op,
        logit_length_pshape.rank().is_dynamic() || DimType::merge(batch_size, batch_size, logit_length_pshape[0]),
        "The first dimension of logit length must be equal to the first dimension ",
        "of the logits. Got: ",
        logit_length_pshape[0],
        " and: ",
        batch_size);

    if (labels_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op,
                              DimType::merge(batch_size, batch_size, labels_pshape[0]),
                              "The first dimension of labels must be equal to the first dimension ",
                              "of the logits and the logit length. Got: ",
                              labels_pshape[0],
                              " and: ",
                              batch_size);

        NODE_VALIDATION_CHECK(op,
                              labels_pshape[1].compatible(time_steps),
                              "The second dimension of labels must be equal to the second dimension ",
                              "of logits. Got: ",
                              labels_pshape[1],
                              " and: ",
                              time_steps);
    }

    NODE_VALIDATION_CHECK(
        op,
        label_length_pshape.rank().is_dynamic() || DimType::merge(batch_size, batch_size, label_length_pshape[0]),
        "The first dimension of label length must be equal to the first dimension ",
        "of the logits, the logit length and labels. Got: ",
        label_length_pshape[0],
        " and: ",
        batch_size);

    return {TRShape{std::move(batch_size)}};
}
}  // namespace v4
}  // namespace op
}  // namespace ov
