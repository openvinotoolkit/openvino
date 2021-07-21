// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"
#include "cldnn/runtime/memory.hpp"
#include <vector>
#include <string>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief This primitive executes custom code provided by the application
/// @details The application is required to provide a function object instance
/// that implements the primitive operation
    struct generic_primitive : public primitive_base<generic_primitive> {
        CLDNN_DECLARE_PRIMITIVE(generic_primitive)

        typedef std::function<event::ptr(const std::vector<event::ptr>& dependent_events,
                                         const std::vector<memory::ptr>& inputs,
                                         const std::vector<memory::ptr>& outputs)>
                execute_function;

        /// @brief Constructs generic_primitive primitive
        /// @param id This primitive id.
        /// @param inputs Input primitive ids.
        /// @param f Callback function to be called by the primitive at execution time
        /// @param output_layout Output layout declared by the primitive
        generic_primitive(const primitive_id& id,
                          const std::vector<primitive_id>& inputs,
                          const execute_function& f,
                          const layout& output_layout)
                : primitive_base(id, {inputs}, output_layout.data_padding),
                  output_layout(output_layout),
                  callback_function(f) {}

        /// @brief The output layout declared by the primitive
        const layout output_layout;
        /// @brief Callback function to be called by the primitive at execution time
        const execute_function callback_function;
    };
/// @}
/// @}
/// @}
}  // namespace cldnn
