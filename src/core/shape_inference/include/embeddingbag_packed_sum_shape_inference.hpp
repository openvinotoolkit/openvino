// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/embeddingbag_packed_base.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace util {

template <class TShape>
std::vector<TShape> shape_infer(const ov::op::util::EmbeddingBagPackedBase* op,
                                const std::vector<TShape>& input_shapes) {
    const auto input_size = input_shapes.size();
    NODE_VALIDATION_CHECK(op, input_size == 2 || input_size == 3);

    static constexpr int EMB_TABLE = 0;
    static constexpr int INDICES = 1;
    static constexpr int PER_SAMPLE_WEIGHTS = 2;

    auto indices_shape = input_shapes[INDICES];
    NODE_VALIDATION_CHECK(op, indices_shape.rank().compatible(2), "INDICES must be 2D.");

    if (input_size == 3) {
        NODE_VALIDATION_CHECK(op,
                              input_shapes[PER_SAMPLE_WEIGHTS].rank().compatible(2),
                              "PER_SAMPLE_WEIGHTS must be 2D.");

        NODE_VALIDATION_CHECK(op,
                              TShape::merge_into(indices_shape, input_shapes[PER_SAMPLE_WEIGHTS]),
                              "INDICES and PER_SAMPLE_WEIGHTS shape must be same.");
    }
    TShape output_shape;
    const auto& emb_table_shape = input_shapes[EMB_TABLE];
    if (emb_table_shape.rank().is_static()) {
        output_shape = emb_table_shape;
        output_shape[0] = indices_shape.rank().is_static() ? indices_shape[0] : Dimension::dynamic();
    } else {
        output_shape = PartialShape::dynamic();
    }
    return {output_shape};
}

template <class TShape>
void shape_infer(const ov::op::util::EmbeddingBagPackedBase* op,
                 const std::vector<TShape>& input_shapes,
                 std::vector<TShape>& output_shapes) {
    output_shapes = shape_infer(op, input_shapes);
}
}  // namespace util
}  // namespace op
}  // namespace ov
