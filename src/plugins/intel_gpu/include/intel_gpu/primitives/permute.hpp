// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"
#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Permutes data in the memory, with respect to provided order.
/// @details Permute order is set as vector with positions meaning corresponding to tensor.
/// Vector values represent dimensions to be permuted in bfyx format. For example: <br>
/// input_dimensions = tensor{ 5, 3, 6, 3 } <br>
/// permute_order = { 2, 3, 1, 0 } <br>
/// output_dimensions = { 6, 3, 3, 5 } <br>
/// <br>
/// When permute_order is { 0, 1, 2, 3 } then input_dimensions = output_dimensions
struct permute : public primitive_base<permute> {
    CLDNN_DECLARE_PRIMITIVE(permute)

    /// @brief Constructs permute primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param permute_order Array of permuted output order in bfyx format.
    permute(const primitive_id& id,
            const input_info& input,
            const std::vector<uint16_t>& permute_order = {},
            const primitive_id& ext_prim_id = "",
            const padding& output_padding = padding())
        : primitive_base(id, {input}, ext_prim_id, {output_padding}), permute_order(permute_order) {}

    /// @brief Array of permuted output order in bfyx format.
    std::vector<uint16_t> permute_order;
};

/// @}
/// @}
/// @}
}  // namespace cldnn
