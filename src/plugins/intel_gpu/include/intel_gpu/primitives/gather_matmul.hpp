// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Gathered GEMM for GatherMatmul pattern.
/// @details
/// Performs per-token weight gathering and matrix multiplication for MoE-style expert execution.
/// Input A has shape [n_activated_experts, batch*seq, hidden_size] with an explicit expert dimension.
/// Weights B have shape [n_all_experts, N, K] and are gathered per-token using indices.
/// Output has shape [n_activated_experts, batch*seq, N].
struct gather_matmul : public primitive_base<gather_matmul> {
    CLDNN_DECLARE_PRIMITIVE(gather_matmul)

    enum BGMInputIdx {
        // required
        INPUT = 0,    // A: [n_activated_experts, batch*seq, hidden_size]
        WEIGHT = 1,   // B: [n_all_experts, N, K] (transposed)
        INDICES = 2,  // [batch*seq, top_k] expert indices
        // optional
        BIAS = 3,  // [n_all_experts, 1, N] or scalar 0
        WEIGHT_SCALE = 4,
        WEIGHT_ZP = 5
    };

    gather_matmul() : primitive_base("", {}) {}

    gather_matmul(const primitive_id& id, const std::vector<input_info>& inputs, bool has_bias, bool has_zp, int32_t n_activated_experts)
        : primitive_base(id, inputs),
          has_bias(has_bias),
          has_zp(has_zp),
          n_activated_experts(n_activated_experts) {}

    bool has_bias = false;
    bool has_zp = false;
    int32_t n_activated_experts = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, has_bias);
        seed = hash_combine(seed, has_zp);
        seed = hash_combine(seed, n_activated_experts);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto rhs_casted = downcast<const gather_matmul>(rhs);
        return has_bias == rhs_casted.has_bias && has_zp == rhs_casted.has_zp && n_activated_experts == rhs_casted.n_activated_experts;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<gather_matmul>::save(ob);
        ob << has_bias;
        ob << has_zp;
        ob << n_activated_experts;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<gather_matmul>::load(ib);
        ib >> has_bias;
        ib >> has_zp;
        ib >> n_activated_experts;
    }
};
}  // namespace cldnn
