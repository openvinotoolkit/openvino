// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"
#include "intel_gpu/runtime/memory.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Provides input layout for a data to be passed later to network.
/// @details This primitive allows to define the layout for input data
/// which will be passed to network before execution.
/// For example, network input images.
/// @note User should call network::set_input_data() for every @p input_layout primitive before network execution.
/// @note @p output_padding property of @p input_layout is ignored - its output layout is always equal to input layout defined during object creation.
/// @sa network::set_input_data(), cldnn::data
struct input_layout : public primitive_base<input_layout> {
    CLDNN_DECLARE_PRIMITIVE(input_layout)

    /// @brief Constructs input layout primitive.
    /// @param id This primitive id.
    /// @param layout Defines layout for the data will be passed to network.
    input_layout(const primitive_id& id, const layout& layout, const primitive_id& ext_prim_id = "")
        : primitive_base(id, {}, ext_prim_id, layout.data_padding), layout(layout) {}

    /// @brief Defines layout for the data will be passed to network.
    mutable cldnn::layout layout;

    void change_layout(const cldnn::layout& new_layout) {
        layout = new_layout;
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
