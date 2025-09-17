// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct moe_gemm : public primitive_base<moe_gemm> {
    CLDNN_DECLARE_PRIMITIVE(moe_gemm)

    enum MoEGemmInputIdx {
        INPUT = 0,
        WEIGHT = 1,
        EXPERTS_IDS = 2,
        INPUT_OFFSET_PER_EXPERT = 3,
        INPUT_TOKENS_LENS = 4,
        BIAS = 5,
        WEIGHT_SCALE = 6,
        WEIGHT_ZP = 7
    };

    moe_gemm() : primitive_base("", {}) {}

    /// @brief Constructs moe_gemm primitive.
    ///
    moe_gemm(const primitive_id& id,
             const input_info& input,
             const input_info& weight,
             const input_info& experts_ids,
             const input_info& inputs_offset_per_expert,
             const input_info& input_tokens_lens,
             const int32_t num_active_experts)
          : primitive_base(id, {input, weight, experts_ids, inputs_offset_per_expert, input_tokens_lens}),
            weight(weight), experts_ids(experts_ids), inputs_offset_per_expert(inputs_offset_per_expert),
            input_tokens_lens(input_tokens_lens), bias(""), weight_scale(""), weight_zp(""),
            num_active_experts(num_active_experts) {}

    moe_gemm(const primitive_id& id,
             const input_info& input,
             const input_info& weight,
             const input_info& experts_ids,
             const input_info& inputs_offset_per_expert,
             const input_info& input_tokens_lens,
             const primitive_id& bias,
             const input_info& weight_scale,
             const input_info& weight_zp,
             const int32_t num_active_experts)
          : primitive_base(id, {input, weight, experts_ids, inputs_offset_per_expert, input_tokens_lens, weight_scale, weight_zp}),
            weight(weight), experts_ids(experts_ids), inputs_offset_per_expert(inputs_offset_per_expert),
            input_tokens_lens(input_tokens_lens), bias(bias), weight_scale(weight_scale), weight_zp(weight_zp),
            num_active_experts(num_active_experts) {}

    moe_gemm(const primitive_id& id,
             const input_info& input,
             const input_info& weight,
             const input_info& experts_ids,
             const input_info& inputs_offset_per_expert,
             const input_info& input_tokens_lens,
             const primitive_id& bias,
             const input_info& weight_scale,
             const primitive_id& weight_zp,
             const int32_t num_active_experts)
          : primitive_base(id, {input, weight, experts_ids, inputs_offset_per_expert, input_tokens_lens, weight_scale}),
            weight(weight), experts_ids(experts_ids), inputs_offset_per_expert(inputs_offset_per_expert),
            input_tokens_lens(input_tokens_lens), bias(bias), weight_scale(weight_scale), weight_zp(weight_zp),
            num_active_experts(num_active_experts) {
            }


    input_info weight;
    input_info experts_ids;
    input_info inputs_offset_per_expert;
    input_info input_tokens_lens;
    input_info bias;
    input_info weight_scale;
    input_info weight_zp;

    bool has_bias = false;
    bool has_weight_scale = false;
    bool has_weight_zp = false;

    int32_t num_active_experts = 0;
    size_t hash() const override {
        return primitive::hash();
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        return true;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe_gemm>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_gemm>::load(ib);
    }
};
}
