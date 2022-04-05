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
struct shuffle_channels : public primitive_base<shuffle_channels> {
    CLDNN_DECLARE_PRIMITIVE(shuffle_channels)

    /// @brief Constructs shuffle_channels primitive.
    /// @param id This primitive id.
    /// @param input Input dictionary primitive id.
    /// @param group The number of groups to split the channel dimension.
    /// @param axis The index of the channel dimension.
    shuffle_channels(const primitive_id& id,
                     const primitive_id& input,
                     const int32_t group,
                     const int32_t axis = 1,
                     const primitive_id& ext_prim_id = "",
                     const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding), group(group), axis(axis) {}

    /// @brief The number of groups to split the channel dimension. This number must evenly divide the channel dimension size.
    int32_t group;
    /// @brief The index of the channel dimension (default is 1).
    int32_t axis;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
