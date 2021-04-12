/*
// Copyright (c) 2021 Intel Corporation
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
struct gather_nd : public primitive_base<gather_nd> {
    CLDNN_DECLARE_PRIMITIVE(gather_nd)

    /// @brief Constructs gather_nd primitive.
    /// @param id This primitive id.
    /// @param data Input data primitive id.
    /// @param indices Input indexes primitive id.
    /// @param indices_rank Rank of indices.
    /// @param batch_dims batch_dims as an attribute of GatherND. Optional.
    gather_nd(const primitive_id& id,
                   const primitive_id& data,
                   const primitive_id& indices,
                   const uint8_t indices_rank,
                   const uint8_t batch_dims = 0,
                   const padding& output_padding = padding())
        : primitive_base(id, {data, indices}, output_padding), indices_rank(indices_rank), batch_dims(batch_dims) {}

    /// @brief GatherND indices_rank
    uint8_t indices_rank;

    /// @brief GatherND batch_dims
    uint8_t batch_dims;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
