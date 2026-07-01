// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Grouped matmul for MoE forward pass (2D x 3D case).
/// @details
/// Computes multiple matrix multiplications where each group processes a subset
/// of the input rows. Matches ov::op::v17::GroupedMatMul 2D x 3D semantics.
///
///   mat_a   : [total_tokens, K]        - rows partitioned by @p offsets
///   mat_b   : [G, N, K]                - per-group weights, stored transposed (B^T)
///   offsets : [G] (s32)                - cumulative end-offsets partitioning mat_a rows
///   output  : [total_tokens, N]
struct grouped_matmul : public primitive_base<grouped_matmul> {
    CLDNN_DECLARE_PRIMITIVE(grouped_matmul)

    enum GroupedMatmulInputIdx {
        INPUT = 0,     // mat_a  : [total_tokens, K]
        WEIGHT = 1,    // mat_b  : [G, N, K]
        OFFSETS = 2,   // offsets: [G] cumulative end-offsets (s32)
    };

    grouped_matmul() : primitive_base("", {}) {}

    grouped_matmul(const primitive_id& id,
                   const std::vector<input_info>& inputs,
                   const data_types output_data_type)
        : primitive_base(id, inputs, 1, {optional_data_type{output_data_type}}) {}

    size_t hash() const override {
        return primitive::hash();
    }

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<grouped_matmul>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<grouped_matmul>::load(ib);
    }
};
}  // namespace cldnn
