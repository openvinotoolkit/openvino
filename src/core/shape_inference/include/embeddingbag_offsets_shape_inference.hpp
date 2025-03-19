// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "embedding_shape_infer_utils.hpp"
#include "openvino/op/embeddingbag_offsets_sum.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace util {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ov::op::util::EmbeddingBagOffsetsBase* op,
                                 const std::vector<TShape>& input_shapes) {
    const auto input_size = input_shapes.size();

    NODE_VALIDATION_CHECK(op, (input_size >= 3 && input_size <= 5));

    static constexpr int EMB_TABLE = 0;
    static constexpr int INDICES = 1;
    static constexpr int OFFSETS = 2;
    static constexpr int DEFAULT_INDEX = 3;
    static constexpr int PER_SAMPLE_WEIGHTS = 4;

    NODE_VALIDATION_CHECK(op, input_shapes[INDICES].rank().compatible(1), "INDICES must be 1D.");
    NODE_VALIDATION_CHECK(op, input_shapes[OFFSETS].rank().compatible(1), "OFFSETS must be 1D.");

    if (input_size >= 4) {
        NODE_VALIDATION_CHECK(op, input_shapes[DEFAULT_INDEX].rank().compatible(0), "DEFAULT_INDEX must be a scalar.");
    }

    if (input_size == 5) {
        NODE_VALIDATION_CHECK(op,
                              input_shapes[PER_SAMPLE_WEIGHTS].rank().compatible(1),
                              "PER_SAMPLE_WEIGHTS must be 1D.");

        NODE_VALIDATION_CHECK(op,
                              input_shapes[INDICES].compatible(input_shapes[PER_SAMPLE_WEIGHTS]),
                              "INDICES and PER_SAMPLE_WEIGHTS shape must be same.");
    }

    return {embedding::out_shape_infer(op, input_shapes[EMB_TABLE], input_shapes[OFFSETS])};
}
}  // namespace util
}  // namespace op
}  // namespace ov
