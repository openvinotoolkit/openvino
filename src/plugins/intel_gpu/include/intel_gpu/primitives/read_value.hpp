// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "primitive.hpp"
#include "intel_gpu/runtime/memory.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Returns value of the variable_id variable.
struct read_value : public primitive_base<read_value> {
    CLDNN_DECLARE_PRIMITIVE(read_value)

    /// @brief Constructs ReadValue primitive.
    /// @param id This primitive id
    /// @param inputs Input parameters ids
    /// @param variable_id Variable id
    /// @param output_layout Memory layout
    read_value(const primitive_id& id,
               const std::vector<primitive_id>& inputs,
               const std::string& variable_id,
               const layout& output_layout)
            : primitive_base(id, inputs, "", {}, optional_data_type{output_layout.data_type}),
              variable_id{variable_id},
              output_layout{output_layout} {}

    std::string variable_id;
    layout output_layout;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
