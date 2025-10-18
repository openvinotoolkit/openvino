// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

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
    /// @param tokens_len_per_expert         tokens len_per_expert
    moe_gather(const primitive_id& id,
              const input_info& data,
              const input_info& tokens_per_expert,
              int32_t num_experts_per_token = 0)
        : primitive_base(id, {data, tokens_per_expert}), num_experts_per_token(num_experts_per_token) {}

    int32_t num_experts_per_token = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, num_experts_per_token);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const moe_gather>(rhs);

        return num_experts_per_token == rhs_casted.num_experts_per_token;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe_gather>::save(ob);
        ob << num_experts_per_token;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_gather>::load(ib);
        ib >> num_experts_per_token;
    }
};
}
