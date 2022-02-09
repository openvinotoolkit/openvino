// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

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

    /// @brief Constructs depth_to_space primitive.
    /// @param id This primitive id.
    /// @param input Input dictionary primitive id.
    /// @param block_size Block size.
    /// @param mode Depth division mode.
    depth_to_space(const primitive_id& id,
                   const primitive_id& input,
                   const size_t block_size,
                   const depth_to_space_mode mode,
                   const primitive_id& ext_prim_id = "",
                   const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding)
        , block_size(block_size)
        , mode(mode) {}

    /// @brief Block size.
    size_t block_size;
    /// @brief depth division mode
    depth_to_space_mode mode;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
