// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
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
struct slice : public primitive_base<slice> {
    CLDNN_DECLARE_PRIMITIVE(slice)

    /// @brief Constructs slice primitive.
    /// @param id This primitive id.
    /// @param inputs List of primitive ids.
    slice(const primitive_id& id,
                  const std::vector<primitive_id>& inputs,
                  const tensor output_shape,
                  const primitive_id& ext_prim_id = "",
                  const padding& output_padding = padding())
        : primitive_base{id, inputs, ext_prim_id, output_padding},
          output_shape {output_shape}
    {}

    tensor output_shape;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
