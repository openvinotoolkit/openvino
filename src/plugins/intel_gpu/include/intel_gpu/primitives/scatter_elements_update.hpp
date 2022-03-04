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
struct scatter_elements_update : public primitive_base<scatter_elements_update> {
    CLDNN_DECLARE_PRIMITIVE(scatter_elements_update)

    enum scatter_elements_update_axis {
        along_b,
        along_f,
        along_x,
        along_y,
        along_z,
        along_w
    };

    /// @brief Constructs scatter_elements_update primitive.
    /// @param id This primitive id.
    /// @param dict Input data primitive id.
    /// @param idx Input indexes primitive id.
    /// @param idupd Input updates primitive id.
    /// @param axis Gathering axis.
    scatter_elements_update(const primitive_id& id,
                            const primitive_id& data,
                            const primitive_id& idx,
                            const primitive_id& idupd,
                            const scatter_elements_update_axis axis,
                            const primitive_id& ext_prim_id = "",
                            const padding& output_padding = padding())
        : primitive_base(id, {data, idx, idupd}, ext_prim_id, output_padding), axis(axis) {}

    /// @brief ScatterElementsUpdate axis
    scatter_elements_update_axis axis;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
