// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/reverse_sequence.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace v0 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ReverseSequence* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);
    using DimType = typename TShape::value_type;

    const auto& data_pshape = input_shapes[0];
    const auto& data_rank = data_pshape.rank();
    const auto& seq_lengths_pshape = input_shapes[1];
    const auto& seq_lengths_rank = seq_lengths_pshape.rank();

    NODE_VALIDATION_CHECK(op,
                          data_rank.is_dynamic() || data_rank.get_length() >= 2,
                          "Data input rank should be equal or greater than 2. Got: ",
                          data_pshape);

    NODE_VALIDATION_CHECK(op,
                          seq_lengths_rank.compatible(1),
                          "Sequence lengths rank must be equal to 1. Got: ",
                          seq_lengths_pshape);
    TRShape output_pshape = data_pshape;
    if (data_rank.is_static() && seq_lengths_rank.is_static()) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        const auto normalized_batch_axis = ov::normalize_axis(op, op->get_origin_batch_axis(), data_rank);
        OPENVINO_SUPPRESS_DEPRECATED_END
        DimType merged_sequence_length;
        NODE_VALIDATION_CHECK(
            op,
            DimType::merge(merged_sequence_length, data_pshape[normalized_batch_axis], seq_lengths_pshape[0]),
            "Sequence lengths input size (",
            seq_lengths_pshape[0],
            ") is not equal to batch axis dimension of data input (",
            data_pshape[normalized_batch_axis],
            ") (argument shape: ",
            data_pshape,
            ", sequence indices shape: ",
            seq_lengths_pshape,
            ").");
        output_pshape[normalized_batch_axis] = merged_sequence_length;
    }

    return {output_pshape};
}
}  // namespace v0
}  // namespace op
}  // namespace ov
