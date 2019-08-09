/*
// Copyright (c) 2016 Intel Corporation
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
#include "generic_layer.h"
#include "api/CPP/primitive.hpp"
#include "api/CPP/memory.hpp"
#include "kernel_selector_helper.h"
#include <vector>

namespace cldnn {

/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Changes how data is ordered in memory. Value type is not changed & all information is preserved.
/// @details Corresponding values are bitwise equal before/after reorder.
/// Also merged with subtraction layer, which can subtract values while doing reordering.
/// NOTE THAT THIS WILL SUBTRACT THE SAME VALUES FROM EACH BATCH.
struct generic_layer : public primitive_base<generic_layer, CLDNN_PRIMITIVE_DESC(generic_layer)> {
    CLDNN_DECLARE_PRIMITIVE(generic_layer)

    /// @brief Constructs generic_layer primitive which takes mean subtract values from another primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_layout Requested memory layout.
    /// @param mean Primitive id to get mean subtract values.
    generic_layer(const primitive_id& id,
                  const primitive_id& input,
                  const layout& output_layout,
                  const kernel_selector::generic_kernel_params& generic_params,
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding), output_layout(output_layout), generic_params(generic_params) {}

    /// @brief Constructs a copy from basic C API @CLDNN_PRIMITIVE_DESC{generic_layer}
    generic_layer(const dto* dto)
        : primitive_base(dto),
          output_layout(dto->output_layout),
          generic_params(*static_cast<const kernel_selector::generic_kernel_params* const>(dto->generic_params)) {}

    /// @brief Requested memory layout.
    layout output_layout;
    const kernel_selector::generic_kernel_params generic_params;

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override { return {}; }

    void update_dto(dto& dto) const override {
        dto.output_layout = output_layout;
        dto.generic_params = &generic_params;
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
