// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/embeddingbag_offsets_sum.hpp>

#include "utils.hpp"
namespace ov {
namespace op {
namespace util {

template <class T>
void shape_infer(const ov::op::util::EmbeddingBagOffsetsBase* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes) {
    const auto input_size = input_shapes.size();

    NODE_VALIDATION_CHECK(op, (input_size >= 3 && input_size <= 5) && output_shapes.size() == 1);

    static constexpr int EMB_TABLE = 0;
    static constexpr int INDICES = 1;
    static constexpr int OFFSETS = 2;
    static constexpr int DEFAULT_INDEX = 3;
    static constexpr int PER_SAMPLE_WEIGHTS = 4;

    NODE_VALIDATION_CHECK(op, input_shapes[INDICES].rank().compatible(1), "INDICES must be 1D");
    NODE_VALIDATION_CHECK(op, input_shapes[OFFSETS].rank().compatible(1), "OFFSETS must be 1D");

    if (input_size >= 4) {
        NODE_VALIDATION_CHECK(op, input_shapes[DEFAULT_INDEX].rank().compatible(0), "DEFAULT_INDEX must be a scalar");
    }

    if (input_size == 5) {
        NODE_VALIDATION_CHECK(op,
                              input_shapes[PER_SAMPLE_WEIGHTS].rank().compatible(1),
                              "PER_SAMPLE_WEIGHTS must be 1D");

        NODE_VALIDATION_CHECK(op,
                              input_shapes[INDICES].compatible(input_shapes[PER_SAMPLE_WEIGHTS]),
                              "INDICES and PER_SAMPLE_WEIGHTS shape must be same");
    }

    const auto& emb_table_shape = input_shapes[EMB_TABLE];
    const auto& offsets_shape = input_shapes[OFFSETS];

    if (emb_table_shape.rank().is_static()) {
        output_shapes[0] = emb_table_shape;
        output_shapes[0][0] = offsets_shape.rank().is_static() ? offsets_shape[0] : Dimension::dynamic();
    } else {
        output_shapes[0] = PartialShape::dynamic();
    }
}
}  // namespace util
}  // namespace op
}  // namespace ov
