/*
// Copyright (c) 2020 Intel Corporation
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
struct scatter_nd_update : public primitive_base<scatter_nd_update> {
    CLDNN_DECLARE_PRIMITIVE(scatter_nd_update)

    /// @brief Constructs scatter_nd_update primitive.
    /// @param id This primitive id.
    /// @param dict Input data primitive id.
    /// @param idx Input indexes primitive id.
    /// @param idupd Input updates primitive id.
    /// @param indices_rank Rank of indices.
    scatter_nd_update(const primitive_id& id,
                   const primitive_id& data,
                   const primitive_id& idx,
                   const primitive_id& idupd,
                   const size_t indices_rank,
                   const padding& output_padding = padding())
        : primitive_base(id, {data, idx, idupd}, output_padding), indices_rank(indices_rank) {}

    /// @brief ScatterNDUpdate indices_rank
    size_t indices_rank;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
