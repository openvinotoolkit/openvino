// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Gathered GEMM for MoE expert execution: per-token weight gather + matmul.
/// A=[n_activated_experts, batch*seq, K], B=[n_all_experts, N, K], out=[n_activated_experts, batch*seq, N].
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
    // Base-3 ternary ("t1") weights: WEIGHT is u8 [n_all_experts, N, K / 64 * 13], each 64 values of a row packed into
    // 13 bytes (5 codes in {0, 1, 2} per byte, scaled base 3), zero point 1. t1_k is the logical K.
    bool weights_t1 = false;
    int64_t t1_k = 0;
    // Binary (u1) weights: WEIGHT is u8 [n_all_experts, N, K / 8], 8 values per byte, most significant bit first,
    // value = (bit - u1_zp) * scale. t1_k is the logical K.
    bool weights_u1 = false;
    float u1_zp = 0.5f;
    // t1/u1 only: WEIGHT_SCALE holds FP8 E4M3 codes (u8) instead of f16.
    bool scales_fp8 = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, has_bias);
        seed = hash_combine(seed, has_zp);
        seed = hash_combine(seed, n_activated_experts);
        seed = hash_combine(seed, weights_t1);
        seed = hash_combine(seed, t1_k);
        seed = hash_combine(seed, weights_u1);
        seed = hash_combine(seed, u1_zp);
        seed = hash_combine(seed, scales_fp8);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs)) {
            return false;
        }
        auto rhs_casted = downcast<const gather_matmul>(rhs);
        return has_bias == rhs_casted.has_bias && has_zp == rhs_casted.has_zp && n_activated_experts == rhs_casted.n_activated_experts &&
               weights_t1 == rhs_casted.weights_t1 && t1_k == rhs_casted.t1_k && weights_u1 == rhs_casted.weights_u1 &&
               u1_zp == rhs_casted.u1_zp && scales_fp8 == rhs_casted.scales_fp8;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<gather_matmul>::save(ob);
        ob << has_bias;
        ob << has_zp;
        ob << n_activated_experts;
        ob << weights_t1;
        ob << t1_k;
        ob << weights_u1;
        ob << u1_zp;
        ob << scales_fp8;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<gather_matmul>::load(ib);
        ib >> has_bias;
        ib >> has_zp;
        ib >> n_activated_experts;
        ib >> weights_t1;
        ib >> t1_k;
        ib >> weights_u1;
        ib >> u1_zp;
        ib >> scales_fp8;
    }
};
}  // namespace cldnn
