// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/embeddingbag_offsets_sum.hpp>

#include "utils.hpp"
namespace ov {
namespace op {
namespace v3 {

template <class TShape>
void shape_infer(const EmbeddingSegmentsSum* op,
                 const std::vector<TShape>& input_shapes,
                 std::vector<TShape>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    output_shapes = shape_infer(op, input_shapes, constant_data);
}

template <class TShape>
std::vector<TShape> shape_infer(
    const EmbeddingSegmentsSum* op,
    const std::vector<TShape>& input_shapes,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    const auto input_size = input_shapes.size();

    NODE_VALIDATION_CHECK(op, input_size >= 4 && input_size <= 6);

    static constexpr int EMB_TABLE = 0;
    static constexpr int INDICES = 1;
    static constexpr int SEGMENT_IDS = 2;
    static constexpr int NUM_SEGMENTS = 3;
    static constexpr int DEFAULT_INDEX = 4;
    static constexpr int PER_SAMPLE_WEIGHTS = 5;

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

    TShape result_shape;
    const auto& emb_table_shape = input_shapes[EMB_TABLE];
    if (emb_table_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op, emb_table_shape.size() > 0, "EMB_TABLE can't be a scalar.");
        result_shape = emb_table_shape;
        TShape segments_value;
        if (get_data_as_shape<TShape>(NUM_SEGMENTS, op, segments_value, constant_data)) {
            result_shape[0] = segments_value[0];
        } else {
            result_shape[0] = Dimension::dynamic();
        }
    } else {
        result_shape = ov::PartialShape::dynamic();
    }
    return {result_shape};
}
}  // namespace v3
}  // namespace op
}  // namespace ov
