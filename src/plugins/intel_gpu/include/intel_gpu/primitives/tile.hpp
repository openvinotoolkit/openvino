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

/// @brief Performs tile operation on input.
/// @details copies the input data n times across chosen axis.
struct tile : public primitive_base<tile> {
    CLDNN_DECLARE_PRIMITIVE(tile)

    /// @brief Constructs tile primitive.
    /// @param id This primitive id.
    /// @param repeats Per-dimension replication factor.
    tile(const primitive_id& id,
         const primitive_id& input,
         const std::vector<int64_t> repeats,
         const primitive_id& ext_prim_id = "",
         const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          repeats(repeats) {}

    /// @brief A per-dimension replication factor
    std::vector<int64_t> repeats;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
