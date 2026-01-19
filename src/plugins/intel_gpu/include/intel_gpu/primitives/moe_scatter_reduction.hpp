// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/op/moe_compressed.hpp"

namespace cldnn {

/// @brief
/// @details
struct moe_scatter_reduction : public primitive_base<moe_scatter_reduction> {
    CLDNN_DECLARE_PRIMITIVE(moe_scatter_reduction)

    moe_scatter_reduction() : primitive_base("", {}) {}

    /// @brief Constructs moe_scatter_reduction primitive.
    ///
    /// @param id                            This primitive id.
    /// @param input                         Input data primitive id.
    /// @param experts_per_token             sorted topk expert id per token
    /// @param expert_weights_per_token      sorted topk expert id weight per token
    /// @param tokens_per_expert             tokens per expert
    /// @param experts_info_offsets          offset of each expert's info from the tokens_per_expert
    /// @param tokens_len_per_expert         tokens len_per_expert
    /// @param experts_ids                   exert_ids actually used
    moe_scatter_reduction(const primitive_id& id,
                          const input_info& data,
                          const input_info& experts_per_token,
                          const input_info& expert_weights_per_token,
                          const input_info& tokens_per_expert,
                          const input_info& experts_info_offsets,
                          const input_info& tokens_len_per_expert,
                          const input_info& experts_ids,
                          const ov::intel_gpu::op::MOECompressed::Config& moe_config)
        : primitive_base(id, {data, experts_per_token, expert_weights_per_token, tokens_per_expert, experts_info_offsets, tokens_len_per_expert, experts_ids}),
          num_active_experts_per_token(static_cast<int32_t>(moe_config.top_k)),
          has_batch_dim(moe_config.has_batch_dim) {}

    int32_t num_active_experts_per_token = 0;
    bool has_batch_dim = true;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, num_active_experts_per_token);
        seed = hash_combine(seed, has_batch_dim);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const moe_scatter_reduction>(rhs);

        return num_active_experts_per_token == rhs_casted.num_active_experts_per_token;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe_scatter_reduction>::save(ob);
        ob << num_active_experts_per_token;
        ob << has_batch_dim;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_scatter_reduction>::load(ib);
        ib >> num_active_experts_per_token;
        ib >> has_batch_dim;
    }
};
}

