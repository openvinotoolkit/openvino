// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/op/moe_compressed.hpp"

namespace cldnn {

/// @brief
/// @details
struct moe_gather : public primitive_base<moe_gather> {
    CLDNN_DECLARE_PRIMITIVE(moe_gather)

    moe_gather() : primitive_base("", {}) {}

    /// @brief Constructs moe_gather primitive.
    ///
    /// @param id                            This primitive id.
    /// @param input                         Input data primitive id.
    /// @param experts_info_offsets          offset of each expert's info from the tokens_per_expert
    /// @param tokens_per_expert             tokens per expert
    moe_gather(const primitive_id& id,
              const input_info& data,
              const input_info& tokens_per_expert,
              const ov::intel_gpu::op::MOECompressed::Config& moe_config)
        : primitive_base(id, {data, tokens_per_expert}), num_experts_per_token(static_cast<int32_t>(moe_config.top_k)), has_batch_dim(moe_config.has_batch_dim) {}

    int32_t num_experts_per_token = 0;
    bool has_batch_dim = true;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, num_experts_per_token);
        seed = hash_combine(seed, has_batch_dim);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const moe_gather>(rhs);
        return num_experts_per_token == rhs_casted.num_experts_per_token && has_batch_dim == rhs_casted.has_batch_dim;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe_gather>::save(ob);
        ob << num_experts_per_token;
        ob << has_batch_dim;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_gather>::load(ib);
        ib >> num_experts_per_token;
        ib >> has_batch_dim;
    }
};
}
