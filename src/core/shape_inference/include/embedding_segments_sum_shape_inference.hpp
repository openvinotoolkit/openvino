// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/embedding_segments_sum.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v3 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const EmbeddingSegmentsSum* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    const auto input_size = input_shapes.size();

    NODE_VALIDATION_CHECK(op, input_size >= 4 && input_size <= 6);

    constexpr size_t EMB_TABLE = 0;
    constexpr size_t INDICES = 1;
    constexpr size_t SEGMENT_IDS = 2;
    constexpr size_t NUM_SEGMENTS = 3;
    constexpr size_t DEFAULT_INDEX = 4;
    constexpr size_t PER_SAMPLE_WEIGHTS = 5;

    NODE_VALIDATION_CHECK(op, input_shapes[INDICES].rank().compatible(1), "INDICES must be 1D.");
    NODE_VALIDATION_CHECK(op, input_shapes[SEGMENT_IDS].rank().compatible(1), "SEGMENT_IDS must be 1D.");
    NODE_VALIDATION_CHECK(op,
                          input_shapes[INDICES].compatible(input_shapes[SEGMENT_IDS]),
                          "INDICES and SEGMENT_IDS shape must be same");

    NODE_VALIDATION_CHECK(op, input_shapes[NUM_SEGMENTS].compatible(TShape{}), "NUM_SEGMENTS must be a scalar.");

    if (input_size >= 5) {
        NODE_VALIDATION_CHECK(op, input_shapes[DEFAULT_INDEX].compatible(TShape{}), "DEFAULT_INDEX must be a scalar.");
    }

    if (input_size == 6) {
        NODE_VALIDATION_CHECK(op,
                              input_shapes[PER_SAMPLE_WEIGHTS].rank().compatible(1),
                              "PER_SAMPLE_WEIGHTS must be 1D.");

        NODE_VALIDATION_CHECK(op,
                              input_shapes[INDICES].compatible(input_shapes[PER_SAMPLE_WEIGHTS]),
                              "INDICES and PER_SAMPLE_WEIGHTS shape must be same.");
    }
    const auto& emb_table_shape = input_shapes[EMB_TABLE];
    auto output_shapes = std::vector<TRShape>{emb_table_shape};
    auto& result_shape = output_shapes[0];
    if (emb_table_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op, emb_table_shape.size() > 0, "EMB_TABLE can't be a scalar.");
        if (auto segments_value = get_input_const_data_as_shape<TRShape>(op, NUM_SEGMENTS, ta)) {
            result_shape[0] = (*segments_value)[0];
        } else {
            result_shape[0] = Dimension::dynamic();
        }
    }
    return output_shapes;
}
}  // namespace v3
}  // namespace op
}  // namespace ov
