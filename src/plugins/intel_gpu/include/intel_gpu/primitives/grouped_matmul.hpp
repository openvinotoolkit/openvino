// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Grouped GEMM for Mixture-of-Experts and batched-uniform use cases.
///
/// Supports two cases that match v17::GroupedMatMul:
///  * Case 1 — 3D x 3D (no offsets, uniform group size):
///      A=[G, M, K], B=[G, N, K] (B pre-transposed)  ->  out=[G, M, N]
///  * Case 2 — 2D x 3D (with cumulative offsets, MoE forward pass):
///      A=[T, K], B=[G, N, K], offsets=[G]            ->  out=[T, N]
struct grouped_matmul : public primitive_base<grouped_matmul> {
    CLDNN_DECLARE_PRIMITIVE(grouped_matmul)

    enum InputIdx {
        INPUT = 0,    // A matrix
        WEIGHT = 1,   // B matrix (pre-transposed: [G, N, K])
        OFFSETS = 2,  // [G] cumulative row-end offsets (2D×3D case only)
        BIAS = 3,     // optional [G, 1, N]
    };

    grouped_matmul() : primitive_base("", {}) {}

    grouped_matmul(const primitive_id& id,
                   const std::vector<input_info>& inputs,
                   bool has_offsets,
                   bool has_bias)
        : primitive_base(id, inputs),
          has_offsets(has_offsets),
          has_bias(has_bias) {}

    bool has_offsets = false;
    bool has_bias = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, has_offsets);
        seed = hash_combine(seed, has_bias);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto rhs_casted = downcast<const grouped_matmul>(rhs);
        return has_offsets == rhs_casted.has_offsets && has_bias == rhs_casted.has_bias;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<grouped_matmul>::save(ob);
        ob << has_offsets;
        ob << has_bias;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<grouped_matmul>::load(ib);
        ib >> has_offsets;
        ib >> has_bias;
    }
};

}  // namespace cldnn
