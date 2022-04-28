// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Roll-7 primitive.
struct roll : public primitive_base<roll> {
    CLDNN_DECLARE_PRIMITIVE(roll)

    /// @brief Constructs roll primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param shift Tensor which specifies the number of places by which the elements are shifted.
    roll(const primitive_id& id,
         const primitive_id& input,
         const tensor& shift,
         const primitive_id& ext_prim_id = {},
         const padding& output_padding = {})
        : primitive_base(id, {input}, ext_prim_id, output_padding),
          shift(shift) {}

    /// @brief Tensor which specifies the number of places by which the elements are shifted.
    tensor shift;
};

/// @}
/// @}
/// @}
}  // namespace cldnn
