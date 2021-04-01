/*
// Copyright (c) 2018-2020 Intel Corporation
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
    /// @param out_shape The shape of tiled tensor.
    tile(const primitive_id& id,
         const primitive_id& input,
         const tensor out_shape,
         const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding), out_shape(out_shape) {}

    /// @brief Shape of the output tensor
    tensor out_shape;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
