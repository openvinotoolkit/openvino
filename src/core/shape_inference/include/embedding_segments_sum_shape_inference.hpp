// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <bitset>
#include <openvino/core/validation_util.hpp>
#include <openvino/op/embeddingbag_offsets_sum.hpp>

#include "utils.hpp"
namespace ov {
namespace op {
namespace v3 {

template <class T>
void shape_infer(const EmbeddingSegmentsSum* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    const auto input_size = input_shapes.size();

    NODE_VALIDATION_CHECK(op, (input_size >= 4 && input_size <= 6) && output_shapes.size() == 1);

    static constexpr int EMB_TABLE = 0;
    static constexpr int INDICES = 1;
    static constexpr int SEGMENT_IDS = 2;
    static constexpr int NUM_SEGMENTS = 3;
    static constexpr int DEFAULT_INDEX = 4;
    static constexpr int PER_SAMPLE_WEIGHTS = 5;

    NODE_VALIDATION_CHECK(op, input_shapes[INDICES].rank().compatible(1), "INDICES must be 1D");
    NODE_VALIDATION_CHECK(op, input_shapes[SEGMENT_IDS].rank().compatible(1), "SEGMENT_IDS must be 1D");
    NODE_VALIDATION_CHECK(op,
                          input_shapes[INDICES].compatible(input_shapes[SEGMENT_IDS]),
                          "INDICES and SEGMENT_IDS shape must be same");

    NODE_VALIDATION_CHECK(op, input_shapes[NUM_SEGMENTS].compatible(T{}), "NUM_SEGMENTS must be a scalar");

    if (input_size >= 5) {
        NODE_VALIDATION_CHECK(op, input_shapes[DEFAULT_INDEX].compatible(T{}), "DEFAULT_INDEX must be a scalar");
    }

    if (input_size == 6) {
        NODE_VALIDATION_CHECK(op,
                              input_shapes[PER_SAMPLE_WEIGHTS].rank().compatible(1),
                              "PER_SAMPLE_WEIGHTS must be 1D");

        NODE_VALIDATION_CHECK(op,
                              input_shapes[INDICES].compatible(input_shapes[PER_SAMPLE_WEIGHTS]),
                              "INDICES and PER_SAMPLE_WEIGHTS shape must be same");
    }

    const auto& emb_table_shape = input_shapes[EMB_TABLE];

    auto& result_shape = output_shapes[0];
    if (emb_table_shape.rank().is_static()) {
        result_shape = emb_table_shape;
        std::vector<int64_t> segments_value;
        if (get_data_as_int64<T>(NUM_SEGMENTS, op, segments_value, constant_data)) {
            result_shape[0] = segments_value[0];
        } else {
            result_shape[0] = Dimension::dynamic();
        }
    } else {
        result_shape = ov::PartialShape::dynamic();
    }
}
}  // namespace v3
}  // namespace op
}  // namespace ov
