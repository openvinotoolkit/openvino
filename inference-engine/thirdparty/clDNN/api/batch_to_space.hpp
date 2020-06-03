/*
// Copyright (c) 2019 Intel Corporation
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

/// @brief
/// @details
struct batch_to_space : public primitive_base<batch_to_space> {
    CLDNN_DECLARE_PRIMITIVE(batch_to_space)

    /// @brief Constructs batch_to_space primitive.
    /// @param id This primitive id.
    /// @param input Input dictionary primitive id.
    /// @param block_shape Array of block sizes.
    /// @param crops_begin Amount to crop from the beginning along each axis of data input.
    /// @param crops_end Amount to crop from the ending along each axis of data input.

    batch_to_space(const primitive_id& id,
                   const primitive_id& input,
                   const primitive_id& block_shape_id,
                   const primitive_id& crops_begin_id,
                   const primitive_id& crops_end_id,
                   const padding& output_padding = padding())
        : primitive_base(id, {input, block_shape_id, crops_begin_id, crops_end_id}, output_padding){}
};
/// @}
/// @}
/// @}
}  // namespace cldnn
