// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "embedding_shape_infer_utils.hpp"
#include "openvino/op/util/embeddingbag_packed_base.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace util {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ov::op::util::EmbeddingBagPackedBase* op,
                                 const std::vector<TShape>& input_shapes) {
    const auto input_size = input_shapes.size();
    NODE_VALIDATION_CHECK(op, input_size == 2 || input_size == 3);

    constexpr size_t EMB_TABLE = 0;
    constexpr size_t INDICES = 1;
    constexpr size_t PER_SAMPLE_WEIGHTS = 2;

    auto indices_shape = TRShape(input_shapes[INDICES]);
    NODE_VALIDATION_CHECK(op, indices_shape.rank().compatible(2), "INDICES must be 2D.");

    if (input_size == 3) {
        NODE_VALIDATION_CHECK(op,
                              input_shapes[PER_SAMPLE_WEIGHTS].rank().compatible(2),
                              "PER_SAMPLE_WEIGHTS must be 2D.");

        NODE_VALIDATION_CHECK(op,
                              TRShape::merge_into(indices_shape, input_shapes[PER_SAMPLE_WEIGHTS]),
                              "INDICES and PER_SAMPLE_WEIGHTS shape must be same.");
    }
    return {embedding::out_shape_infer(op, input_shapes[EMB_TABLE], TShape(std::move(indices_shape)))};
}
}  // namespace util
}  // namespace op
}  // namespace ov
