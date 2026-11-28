// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/op/moe.hpp"
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct moe_mask_gen : public primitive_base<moe_mask_gen> {
    CLDNN_DECLARE_PRIMITIVE(moe_mask_gen)

    using expert_type = ov::op::internal::MOE::Expert_type;

    // GEMM2_BIAS_SWIGLU_CLAMP outputs
    enum MoEMaskGenOutputIdx { TOKENS_PER_EXPERT = 0, EXPERTS_INFO_START_IDX = 1, EXPERTS_ID = 2, TOKENS_LENS_PER_EXPERT = 3, NUM_ACTUALLY_USED_EXPERTS = 4 };
    // GEMM3_SWIGLU outputs
    enum MoEMaskGen3GemmOutputIdx { TOKEN_INDEX = 0, EXPERT_END_OFFSETS = 1, ROW_LUT = 2 };

    moe_mask_gen() : primitive_base("", {}) {}

    /// @brief Constructs moe_mask_gen primitive.
    ///
    /// @param id                   This primitive id.
    /// @param router_idx           output of router (i.e., idx output of topk)
    /// @param moe_type             GEMM2_BIAS_SWIGLU_CLAMP (moe_gemm) or GEMM3_SWIGLU (moe_3gemm_fused_compressed, GroupedMatMul) outputs
    ///
    /// GEMM2_BIAS_SWIGLU_CLAMP outputs (MoEMaskGenOutputIdx):
    /// @param output0 :            token of each sorted row
    /// @param output1 :            inclusive end row of each expert
    /// @param output2 :            experts' ids actually used
    /// @param output3 :            tokens' length per expert
    /// @param output4 :            number of actually used experts
    ///
    /// GEMM3_SWIGLU outputs (MoEMaskGen3GemmOutputIdx):
    /// @param output0 :            token of each sorted row
    /// @param output1 :            inclusive end row of each expert
    /// @param output2 :            sorted row of each (token, top-k slot) pair (row_lut)

    moe_mask_gen(const primitive_id& id,
                 const input_info& router_idx,
                 const int32_t num_total_experts,
                 const int32_t num_experts_per_token,
                 const expert_type moe_type = expert_type::GEMM2_BIAS_SWIGLU_CLAMP)
        : primitive_base(id, {router_idx}, moe_type == expert_type::GEMM3_SWIGLU ? 3 : 5),
          num_total_experts(num_total_experts),
          num_experts_per_token(num_experts_per_token),
          moe_type(moe_type) {}

    int32_t num_total_experts = 0;
    int32_t num_experts_per_token = 0;
    expert_type moe_type = expert_type::GEMM2_BIAS_SWIGLU_CLAMP;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, num_total_experts);
        seed = hash_combine(seed, num_experts_per_token);
        seed = hash_combine(seed, static_cast<int>(moe_type));
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs)) {
            return false;
        }
        if (const auto* rhs_casted = dynamic_cast<const moe_mask_gen*>(&rhs)) {
            return num_total_experts == rhs_casted->num_total_experts && num_experts_per_token == rhs_casted->num_experts_per_token &&
                   moe_type == rhs_casted->moe_type;
        }
        return false;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe_mask_gen>::save(ob);
        ob << num_total_experts;
        ob << num_experts_per_token;
        ob << make_data(&moe_type, sizeof(moe_type));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_mask_gen>::load(ib);
        ib >> num_total_experts;
        ib >> num_experts_per_token;
        ib >> make_data(&moe_type, sizeof(moe_type));
    }
};

struct moe_mask_gen_reshape : public primitive_base<moe_mask_gen_reshape> {
    CLDNN_DECLARE_PRIMITIVE(moe_mask_gen_reshape)

    enum MoEMaskGenReshapeOutputIdx { TOKENS_PER_EXPERT = 0, EXPERTS_INFO_START_IDX = 1, EXPERTS_ID = 2, TOKENS_LENS_PER_EXPERT = 3 };

    moe_mask_gen_reshape() : primitive_base("", {}) {}

    /// @brief Constructs moe_mask_gen_reshape primitive.
    ///
    /// @param id                   This primitive id.
    /// @param input0 :             output0 of moe_mask_gen
    /// @param input1 :             output1 of moe_mask_gen
    /// @param input2 :             output2 of moe_mask_gen
    /// @param input3 :             output3 of moe_mask_gen
    /// @param input4 :             output4 of moe_mask_gen
    ///
    /// @param output0 :            output0 of moe_mask_gen reshaped for num of actually used experts
    /// @param output1 :            output1 of moe_mask_gen reshaped for num of actually used experts
    /// @param output2 :            output2 of moe_mask_gen reshaped for num of actually used experts
    /// @param output3 :            output3 of moe_mask_gen reshaped for num of actually used experts

    moe_mask_gen_reshape(const primitive_id& id,
                         const input_info& tokens_per_expert,
                         const input_info& experts_info_start_idx,
                         const input_info& experts_id,
                         const input_info& tokens_lens_per_expert,
                         const input_info& num_actual_used_experts)
        : primitive_base(id, {tokens_per_expert, experts_info_start_idx, experts_id, tokens_lens_per_expert, num_actual_used_experts}, 4) {}

    size_t hash() const override {
        return primitive::hash();
    }

    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs);
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe_mask_gen_reshape>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_mask_gen_reshape>::load(ib);
    }
};

}  // namespace cldnn
