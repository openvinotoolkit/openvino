// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief mode for the @ref depth_to_space primitive.
enum class depth_to_space_mode : int32_t {
    /// @brief the input depth is divided to [block_size, ..., block_size, new_depth].
    blocks_first,
    /// @brief the input depth is divided to [new_depth, block_size, ..., block_size]
    depth_first
};

/// @brief
/// @details
struct depth_to_space : public primitive_base<depth_to_space> {
    CLDNN_DECLARE_PRIMITIVE(depth_to_space)

    depth_to_space() : primitive_base("", {}) {}

    /// @brief Constructs depth_to_space primitive.
    /// @param id This primitive id.
    /// @param input Input dictionary primitive id.
    /// @param block_size Block size.
    /// @param mode Depth division mode.
    depth_to_space(const primitive_id& id,
                   const input_info& input,
                   const size_t block_size,
                   const depth_to_space_mode mode)
        : primitive_base(id, {input})
        , block_size(block_size)
        , mode(mode) {}

    /// @brief Block size.
    size_t block_size = 0;
    /// @brief depth division mode
    depth_to_space_mode mode = depth_to_space_mode::blocks_first;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, block_size);
        seed = hash_combine(seed, mode);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const depth_to_space>(rhs);

        return block_size == rhs_casted.block_size &&
               mode == rhs_casted.mode;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<depth_to_space>::save(ob);
        ob << block_size;
        ob << make_data(&mode, sizeof(depth_to_space_mode));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<depth_to_space>::load(ib);
        ib >> block_size;
        ib >> make_data(&mode, sizeof(depth_to_space_mode));
    }
};
}  // namespace cldnn
