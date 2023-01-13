// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct scatter_update : public primitive_base<scatter_update> {
    CLDNN_DECLARE_PRIMITIVE(scatter_update)

    enum scatter_update_axis {
        along_b,
        along_f,
        along_x,
        along_y,
        along_z,
        along_w
    };

    /// @brief Constructs scatter_update primitive.
    /// @param id This primitive id.
    /// @param dict Input dictionary primitive id.
    /// @param idx Input indexes primitive id.
    /// @param idupd Input updates primitive id.
    /// @param axis Gathering axis.
    scatter_update(const primitive_id& id,
                   const input_info& dict,
                   const input_info& idx,
                   const input_info& idupd,
                   const int64_t axis,
                   const padding& output_padding = padding())
        : primitive_base(id, {dict, idx, idupd}, {output_padding}), axis(axis) {}

    /// @brief ScatterUpdate axis
    int64_t axis;
};
}  // namespace cldnn
