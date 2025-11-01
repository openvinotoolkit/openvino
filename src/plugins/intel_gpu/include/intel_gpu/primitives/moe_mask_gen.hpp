// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct moe_mask_gen : public primitive_base<moe_mask_gen> {
    CLDNN_DECLARE_PRIMITIVE(moe_mask_gen)

    enum MoEMaskGenOutputIdx {
        TOKENS_PER_EXPERT         = 0,
        EXPERTS_INFO_START_IDX    = 1,
        EXPERTS_ID                = 2,
        TOKENS_LENS_PER_EXPERT    = 3,
        NUM_ACTUALLY_USED_EXPERTS = 4
    };

    moe_mask_gen() : primitive_base("", {}) {}

    /// @brief Constructs moe_mask_gen primitive.
    ///
    /// @param id                   This primitive id.
    /// @param router_idx           TopK output:1
    /// @param output0 :            tokens_per_expert
    /// @param output1 :            experts_info_start_idx
    /// @param output2 :            experts_id
    /// @param output3 :            tokens_lens_per_expert
    /// @param output4 :            num actually used experts

    moe_mask_gen(const primitive_id& id,
              const input_info& router_idx,
              const int32_t num_total_experts,
              const int32_t num_experts_per_token)
        : primitive_base(id, {router_idx}, 5),
          num_total_experts(num_total_experts),
          num_experts_per_token(num_experts_per_token) {}

    int32_t num_total_experts = 0;
    int32_t num_experts_per_token = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, num_total_experts);
        seed = hash_combine(seed, num_experts_per_token);
        return primitive::hash();
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        return true;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe_mask_gen>::save(ob);
        ob << num_total_experts;
        ob << num_experts_per_token;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_mask_gen>::load(ib);
        ib >> num_total_experts;
        ib >> num_experts_per_token;
    }
};

struct moe_mask_gen_reshape : public primitive_base<moe_mask_gen_reshape> {
    CLDNN_DECLARE_PRIMITIVE(moe_mask_gen_reshape)

    enum MoEMaskGenReshapeOutputIdx {
        TOKENS_PER_EXPERT      = 0,
        EXPERTS_INFO_START_IDX = 1,
        EXPERTS_ID             = 2,
        TOKENS_LENS_PER_EXPERT = 3
    };

    moe_mask_gen_reshape() : primitive_base("", {}) {}

    /// @brief Constructs moe_mask_gen_reshape primitive.
    ///
    /// @param id                   This primitive id.
    /// @param input0 :             num_actually_used_experts
    /// @param input1 :             tokens_per_expert
    /// @param input2 :             experts_info_start_idx
    /// @param input3 :             experts_id
    /// @param input4 :             tokens_lens_per_expert
    ///
    /// @param output0 :            tokens_per_expert
    /// @param output1 :            experts_info_start_idx
    /// @param output2 :            experts_id
    /// @param output3 :            tokens_lens_per_expert

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
        if (!compare_common_params(rhs))
            return false;
        return true;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe_mask_gen_reshape>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_mask_gen_reshape>::load(ib);
    }
};

}
