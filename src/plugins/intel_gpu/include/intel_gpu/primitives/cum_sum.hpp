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
            const primitive_id& ext_prim_id = "",
            const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, output_padding), axis(axis), exclusive(exclusive), reverse(reverse)
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
