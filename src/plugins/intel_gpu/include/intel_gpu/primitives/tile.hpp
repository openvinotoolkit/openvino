// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Performs tile operation on input.
/// @details copies the input data n times across chosen axis.
struct tile : public primitive_base<tile> {
    CLDNN_DECLARE_PRIMITIVE(tile)

    tile() : primitive_base("", {}) {}

    /// @brief Constructs tile primitive with static input.
    /// @param id This primitive id.
    /// @param repeats Per-dimension replication factor.
    tile(const primitive_id& id,
         const input_info& input,
         const std::vector<int64_t> repeats)
        : primitive_base(id, {input}),
          repeats(repeats) {}

    // @brief Constructs tile primitive with dynamic input.
    tile(const primitive_id& id,
         const input_info& input,
         const input_info& repeats_id)
        : primitive_base(id, {input, repeats_id}),
          repeats({}) {}

    /// @brief A per-dimension replication factor
    std::vector<int64_t> repeats;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_range(seed, repeats.begin(), repeats.end());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const tile>(rhs);

        return repeats == rhs_casted.repeats;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<tile>::save(ob);
        ob << repeats;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<tile>::load(ib);
        ib >> repeats;
    }
};
}  // namespace cldnn
