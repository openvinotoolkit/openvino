/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

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
    /// @param axis Tiling axis
    /// @param tiles Tiles number across an axis
    tile(const primitive_id& id,
         const primitive_id& input,
         const tile_axis axis,
         const int tiles,
         const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding), axis(axis), tiles(tiles) {}

    /// @brief Tiling axis
    tile_axis axis;
    /// @brief Tiles number across an axis
    int tiles;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
