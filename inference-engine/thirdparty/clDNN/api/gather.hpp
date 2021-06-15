// Copyright (C) 2018-2021 Intel Corporation
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
struct gather : public primitive_base<gather> {
    CLDNN_DECLARE_PRIMITIVE(gather)

    enum gather_axis {
        along_b,
        along_f,
        along_x,
        along_y,
        along_z,
        along_w
    };

    /// @brief Constructs gather primitive.
    /// @param id This primitive id.
    /// @param dict Input dictionary primitive id.
    /// @param idx Input indexes primitive id.
    /// @param axis Gathering axis.
    /// @param output_shape Output shape.
    /// @param batch_dim Batch_dim
    gather(const primitive_id& id,
           const primitive_id& dict,
           const primitive_id& idx,
           const gather_axis axis,
           const format& output_format,
           const tensor& output_shape,
           const int64_t batch_dim = 0,
           const padding& output_padding = padding())
        : primitive_base(id, {dict, idx}, output_padding), axis(axis), output_format(output_format), output_shape(output_shape), batch_dim(batch_dim) {}

    /// @brief Gathering axis
    gather_axis axis;
    /// @brief Gather output format
    format output_format;
    /// @brief Gather output shape
    tensor output_shape;
    /// @brief Gathering batch_dim
    int64_t batch_dim;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
