// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Performs tile operation on input.
/// @details copies the input data n times across chosen axis.
struct tile : public primitive_base<tile> {
    CLDNN_DECLARE_PRIMITIVE(tile)

    /// @brief Constructs tile primitive with static input.
    /// @param id This primitive id.
    /// @param repeats Per-dimension replication factor.
    tile(const primitive_id& id,
         const input_info& input,
         const std::vector<int64_t> repeats,
         const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}),
          repeats(repeats) {}

    // @brief Constructs tile primitive with dynamic input.
    tile(const primitive_id& id,
         const input_info& input,
         const input_info& repeats_id,
         const padding& output_padding = padding())
        : primitive_base(id, {input, repeats_id}, {output_padding}),
          repeats({}) {}

    /// @brief A per-dimension replication factor
    std::vector<int64_t> repeats;

    size_t hash() const override {
        if (!seed) {
            seed = hash_range(seed, repeats.begin(), repeats.end());
        }
        return seed;
    }
};
}  // namespace cldnn
