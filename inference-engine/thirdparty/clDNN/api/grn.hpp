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

/// @brief Global Response Normalization primitive.
struct grn : public primitive_base<grn> {
    CLDNN_DECLARE_PRIMITIVE(grn)

    /// @brief Constructs grn primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param bias Bias value for whole output tensor.
    grn(const primitive_id& id,
        const primitive_id& input,
        const float bias,
        const data_types data_type,
        const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding, optional_data_type{ data_type }),
        bias(bias)
    {}

    /// @brief Bias value for whole output tensor.
    float bias;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
