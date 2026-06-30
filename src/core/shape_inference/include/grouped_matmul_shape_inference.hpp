// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/grouped_matmul.hpp"
#include "utils.hpp"

namespace ov::op::v17 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const GroupedMatMul* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    const auto num_inputs = input_shapes.size();
    NODE_VALIDATION_CHECK(op, num_inputs == 2 || num_inputs == 3);

    const auto& mat_a_shape = input_shapes[0];
    const auto& mat_b_shape = input_shapes[1];

    // Handle fully dynamic case
    if (mat_a_shape.rank().is_dynamic() || mat_b_shape.rank().is_dynamic()) {
        return {PartialShape::dynamic()};
    }

    const auto mat_a_rank = mat_a_shape.size();
    const auto mat_b_rank = mat_b_shape.size();

    using DimType = typename TShape::value_type;

    // Case: 3D × 3D (batched, uniform group sizes) - no offsets needed
    if (mat_a_rank == 3 && mat_b_rank == 3) {
        const auto g_a = mat_a_shape[0];
        const auto m = mat_a_shape[1];
        const auto k_a = mat_a_shape[2];
        const auto g_b = mat_b_shape[0];
        const auto n = mat_b_shape[1];
        const auto k_b = mat_b_shape[2];

        auto merged_g = DimType();
        auto merged_k = DimType();
        NODE_VALIDATION_CHECK(op,
                              DimType::merge(merged_g, g_a, g_b),
                              "GroupedMatMul 3D×3D: batch dimension mismatch mat_a: ",
                              g_a,
                              ", mat_b: ",
                              g_b);
        NODE_VALIDATION_CHECK(op,
                              DimType::merge(merged_k, k_a, k_b),
                              "GroupedMatMul 3D×3D: inner dimension mismatch mat_a: ",
                              k_a,
                              ", mat_b: ",
                              k_b);

        return {TRShape{merged_g, m, n}};
    }

    // Case: 2D × 3D (MoE forward pass) - requires offsets
    if (mat_a_rank == 2 && mat_b_rank == 3) {
        NODE_VALIDATION_CHECK(op, num_inputs == 3, "GroupedMatMul 2D×3D case requires offsets input.");

        const auto& offsets_shape = input_shapes[2];
        NODE_VALIDATION_CHECK(op,
                              offsets_shape.rank().is_dynamic() || offsets_shape.size() == 1,
                              "GroupedMatMul offsets must be 1D tensor.");

        const auto total_rows = mat_a_shape[0];
        const auto k_a = mat_a_shape[1];
        const auto g = mat_b_shape[0];
        const auto n = mat_b_shape[1];
        const auto k_b = mat_b_shape[2];

        auto merged_k = DimType();
        NODE_VALIDATION_CHECK(op,
                              DimType::merge(merged_k, k_a, k_b),
                              "GroupedMatMul 2D×3D: inner dimension mismatch mat_a: ",
                              k_a,
                              ", mat_b: ",
                              k_b);

        // Output has same number of rows as mat_a
        return {TRShape{total_rows, n}};
    }

    NODE_VALIDATION_CHECK(op,
                          false,
                          "GroupedMatMul unsupported combination: mat_a ",
                          mat_a_rank,
                          "D × mat_b ",
                          mat_b_rank,
                          "D. Supported: 2D×3D, 3D×3D.");

    return {PartialShape::dynamic()};
}

}  // namespace ov::op::v17
