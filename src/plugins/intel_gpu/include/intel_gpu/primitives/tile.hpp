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

    enum tile_axis {
        along_b,
        along_f,
        along_x,
        along_y,
        along_z
    };

    /// @brief Constructs tile primitive.
    /// @param id This primitive id.
    /// @param out_shape The shape of tiled tensor.
    tile(const primitive_id& id,
         const input_info& input,
         const tensor out_shape,
         const primitive_id& ext_prim_id = "",
         const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, {output_padding}), out_shape(out_shape) {}

    /// @brief Shape of the output tensor
    tensor out_shape;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
