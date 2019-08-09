/*
// Copyright (c) 2019 Intel Corporation
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
#include "../C/gather.h"
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
struct gather : public primitive_base<gather, CLDNN_PRIMITIVE_DESC(gather)> {
    CLDNN_DECLARE_PRIMITIVE(gather)

    enum gather_axis {
        along_b = cldnn_gather_along_b,
        along_f = cldnn_gather_along_f,
        along_x = cldnn_gather_along_x,
        along_y = cldnn_gather_along_y
    };

    /// @brief Constructs gather primitive.
    /// @param id This primitive id.
    /// @param dict Input dictionary primitive id.
    /// @param idx Input indexes primitive id.
    /// @param axis Gathering axis.
    /// @param output_shape Output shape.
    gather(const primitive_id& id,
           const primitive_id& dict,
           const primitive_id& idx,
           const gather_axis axis,
           const tensor& output_shape,
           const padding& output_padding = padding())
        : primitive_base(id, {dict, idx}, output_padding), axis(axis), output_shape(output_shape) {}

    /// @brief Constructs a copy from C API @CLDNN_PRIMITIVE_DESC{gather}
    gather(const dto* dto)
        : primitive_base(dto), axis(static_cast<gather_axis>(dto->axis)), output_shape(dto->output_shape) {}

    /// @brief Gathering axis
    gather_axis axis;
    /// @brief Gathering input shape
    tensor output_shape;

protected:
    void update_dto(dto& dto) const override {
        dto.axis = static_cast<cldnn_gather_axis>(axis);
        dto.output_shape = output_shape;
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
