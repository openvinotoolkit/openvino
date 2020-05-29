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

struct cum_sum : public primitive_base<cum_sum> {
    CLDNN_DECLARE_PRIMITIVE(cum_sum)

    enum cum_sum_axis {
        along_b,
        along_f,
        along_x,
        along_y,
        along_z,
        along_w
    };

    /// @brief Constructs cum_sum primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param axis Scalar axis.
    /// @param exclusive If set to true then the top element is not included in sum.
    /// @param reverse If set to true will perform the sums in reverse direction.
    cum_sum(const primitive_id& id,
            const primitive_id& input,
            const cum_sum_axis axis = along_b,
            const bool exclusive = false,
            const bool reverse = false,
            const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding), axis(axis), exclusive(exclusive), reverse(reverse)
    {}

    /// @brief Scalar axis.
    cum_sum_axis axis;
    /// @brief If set to true then the top element is not included in sum.
    bool exclusive;
    /// @brief If set to true will perform the sums in reverse direction.
    bool reverse;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
