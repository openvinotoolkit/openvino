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
    NODE_VALIDATION_CHECK(op, num_inputs == 4, "GroupedMatMul expects exactly 4 inputs.");

    const auto& mat_a_shape = input_shapes[0];
    const auto& mat_b_shape = input_shapes[1];
    const auto& offsets_shape = input_shapes[2];
    const auto& bias_shape = input_shapes[3];

    const auto mat_a_rank = mat_a_shape.rank();
    const auto mat_b_rank = mat_b_shape.rank();

    // Handle fully dynamic case
    if (mat_a_rank.is_dynamic() || mat_b_rank.is_dynamic()) {
        return {PartialShape::dynamic()};
    }

    const auto a_ndim = mat_a_shape.size();
    const auto b_ndim = mat_b_shape.size();

    // Helper: detect Shape{0} — the "empty / not applicable" placeholder
    auto is_empty = [](const TShape& s) -> bool {
        if (s.rank().is_dynamic())
            return false;
        return s.rank().get_length() == 1 && s[0].is_static() && s[0].get_length() == 0;
    };

    const bool offsets_is_empty = is_empty(offsets_shape);
    const bool bias_is_empty = is_empty(bias_shape);

    using DimType = typename TShape::value_type;

    // Case 2: 3D × 3D (batched, uniform group sizes) - offsets are Shape{0}
    if (a_ndim == 3 && b_ndim == 3) {
        const auto G_a = mat_a_shape[0];
        const auto M = mat_a_shape[1];
        const auto K_a = mat_a_shape[2];
        const auto G_b = mat_b_shape[0];
        const auto N = mat_b_shape[1];
        const auto K_b = mat_b_shape[2];

        auto merged_G = DimType();
        auto merged_K = DimType();
        NODE_VALIDATION_CHECK(op,
                              DimType::merge(merged_G, G_a, G_b) || G_a.is_dynamic() || G_b.is_dynamic(),
                              "GroupedMatMul 3D×3D: batch dimension mismatch (mat_a: ",
                              G_a,
                              ", mat_b: ",
                              G_b,
                              ").");
        NODE_VALIDATION_CHECK(op,
                              DimType::merge(merged_K, K_a, K_b) || K_a.is_dynamic() || K_b.is_dynamic(),
                              "GroupedMatMul 3D×3D: inner dimension mismatch (mat_a: ",
                              K_a,
                              ", mat_b: ",
                              K_b,
                              ").");

        if (!bias_is_empty && bias_shape.rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  bias_shape.size() == 2,
                                  "GroupedMatMul bias must be 2D [G, N], got rank: ",
                                  bias_shape.size());
        }

        return {TRShape{merged_G, M, N}};
    }

    // Case 1: 2D × 3D (MoE forward pass) - requires non-empty offsets
    if (a_ndim == 2 && b_ndim == 3) {
        NODE_VALIDATION_CHECK(op,
                              !offsets_is_empty,
                              "GroupedMatMul 2D×3D case requires offsets input.");

        if (!offsets_is_empty) {
            NODE_VALIDATION_CHECK(op,
                                  offsets_shape.rank().is_dynamic() || offsets_shape.size() == 1,
                                  "GroupedMatMul offsets must be 1D tensor.");
        }

        const auto total_rows = mat_a_shape[0];
        const auto K_a = mat_a_shape[1];
        const auto G = mat_b_shape[0];
        const auto N = mat_b_shape[1];
        const auto K_b = mat_b_shape[2];

        auto merged_K = DimType();
        NODE_VALIDATION_CHECK(op,
                              DimType::merge(merged_K, K_a, K_b) || K_a.is_dynamic() || K_b.is_dynamic(),
                              "GroupedMatMul 2D×3D: inner dimension mismatch (mat_a: ",
                              K_a,
                              ", mat_b: ",
                              K_b,
                              ").");

        if (!bias_is_empty && bias_shape.rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  bias_shape.size() == 2,
                                  "GroupedMatMul bias must be 2D [G, N], got rank: ",
                                  bias_shape.size());
        }

        // Output has same number of rows as mat_a
        return {TRShape{total_rows, N}};
    }

    // Case 3: 2D × 2D (MoE weight gradient) - requires non-empty offsets
    if (a_ndim == 2 && b_ndim == 2) {
        NODE_VALIDATION_CHECK(op,
                              !offsets_is_empty,
                              "GroupedMatMul 2D×2D case requires offsets input.");

        const auto& off_shape = offsets_shape;
        NODE_VALIDATION_CHECK(op,
                              off_shape.rank().is_dynamic() || off_shape.size() == 1,
                              "GroupedMatMul offsets must be 1D tensor.");

        const auto K = mat_a_shape[0];
        const auto total_tokens_a = mat_a_shape[1];
        const auto N = mat_b_shape[0];
        const auto total_tokens_b = mat_b_shape[1];

        auto merged_tokens = DimType();
        NODE_VALIDATION_CHECK(op,
                              DimType::merge(merged_tokens, total_tokens_a, total_tokens_b) ||
                                  total_tokens_a.is_dynamic() || total_tokens_b.is_dynamic(),
                              "GroupedMatMul 2D×2D: shared dimension mismatch (mat_a: ",
                              total_tokens_a,
                              ", mat_b: ",
                              total_tokens_b,
                              ").");

        if (!bias_is_empty && bias_shape.rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  bias_shape.size() == 2,
                                  "GroupedMatMul bias must be 2D [G, N], got rank: ",
                                  bias_shape.size());
        }

        // Output shape is (G, K, N) where G is determined by offsets
        auto G = DimType();
        if (off_shape.rank().is_static() && off_shape[0].is_static()) {
            G = off_shape[0];
        } else {
            G = Dimension::dynamic();
        }

        return {TRShape{G, K, N}};
    }

    NODE_VALIDATION_CHECK(op,
                          false,
                          "GroupedMatMul unsupported combination: mat_a ",
                          a_ndim,
                          "D × mat_b ",
                          b_ndim,
                          "D. Supported: 2D×3D, 3D×3D, 2D×2D.");

    return {PartialShape::dynamic()};
}

}  // namespace ov::op::v17
