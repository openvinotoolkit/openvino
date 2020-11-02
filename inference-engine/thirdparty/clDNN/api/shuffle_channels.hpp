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
                     const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding), group(group), axis(axis) {}

    /// @brief The number of groups to split the channel dimension. This number must evenly divide the channel dimension size.
    int32_t group;
    /// @brief The index of the channel dimension (default is 1).
    int32_t axis;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
