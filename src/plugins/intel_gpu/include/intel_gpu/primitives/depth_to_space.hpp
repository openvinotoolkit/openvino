// Copyright (C) 2018-2026 Intel Corporation
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
    depth_first,
    /// @brief Grouped channel repetition followed by anisotropic depth-to-space.
    grouped_depth_first
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

    depth_to_space(const primitive_id& id,
                   const input_info& input,
                   size_t factor_t,
                   size_t factor_s,
                   size_t output_channels,
                   size_t crop_begin_t)
        : primitive_base(id, {input})
        , mode(depth_to_space_mode::grouped_depth_first)
        , factor_t(factor_t)
        , factor_s(factor_s)
        , output_channels(output_channels)
        , crop_begin_t(crop_begin_t) {}

    /// @brief Block size.
    size_t block_size = 0;
    /// @brief depth division mode
    depth_to_space_mode mode = depth_to_space_mode::blocks_first;
    size_t factor_t = 1;
    size_t factor_s = 1;
    size_t output_channels = 0;
    size_t crop_begin_t = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, block_size);
        seed = hash_combine(seed, mode);
        seed = hash_combine(seed, factor_t);
        seed = hash_combine(seed, factor_s);
        seed = hash_combine(seed, output_channels);
        seed = hash_combine(seed, crop_begin_t);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs)) {
            return false;
        }

        auto rhs_casted = downcast<const depth_to_space>(rhs);

         return block_size == rhs_casted.block_size &&
             mode == rhs_casted.mode &&
             factor_t == rhs_casted.factor_t && factor_s == rhs_casted.factor_s &&
               output_channels == rhs_casted.output_channels && crop_begin_t == rhs_casted.crop_begin_t;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<depth_to_space>::save(ob);
        ob << block_size;
        ob << make_data(&mode, sizeof(depth_to_space_mode));
        ob << factor_t;
        ob << factor_s;
        ob << output_channels;
        ob << crop_begin_t;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<depth_to_space>::load(ib);
        ib >> block_size;
        ib >> make_data(&mode, sizeof(depth_to_space_mode));
        ib >> factor_t;
        ib >> factor_s;
        ib >> output_channels;
        ib >> crop_begin_t;
    }
};
}  // namespace cldnn
