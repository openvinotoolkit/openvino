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
                      const input_info& data,
                      const input_info& idx,
                      const input_info& idupd,
                      const size_t indices_rank,
                      const primitive_id& ext_prim_id = "",
                      const padding& output_padding = padding())
        : primitive_base(id, {data, idx, idupd}, ext_prim_id, {output_padding}), indices_rank(indices_rank) {}

    /// @brief ScatterNDUpdate indices_rank
    size_t indices_rank;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
