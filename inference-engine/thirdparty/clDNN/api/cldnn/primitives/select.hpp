/*
// Copyright (c) 2018 Intel Corporation
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

/// @brief Performs elementwise select operation on two input primitives with selector primitive (mask)
/// @notes
/// - format of both inputs has to be the same
/// - if broadcast_type=="numpy", both inputs have to be broadcastable to each other in a two-way
/// (multidirectional) sense and mask input has to be broadcastable in a one-way (unidirectional)
/// sense to the result of this two-way (multidirectional) broadcasting of both inputs to each other.
/// - if broadcast_type=="none", all inputs (including mask) must have the same shape.
///
/// If broadcast_type=="numpy", broadcasting follow numpy-style (ONNX) rules described here:
/// https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md)
struct select : public primitive_base<select> {
    CLDNN_DECLARE_PRIMITIVE(select)

    /// @brief Constructs select primitive.
    /// @param id This primitive id.
    /// @param mask Input primitive id with values needed for select computation.
    /// @param input Input primitive id.
    /// @param input2 Second input primitive id.
    /// @param output_padding Output data padding information.
    /// @param broadcast_type String which determines broadcasting type:
    /// "numpy" means that numpy-tyle (ONNX) broadcasting is allowed,
    /// "none" means that all inputs need to have the same shape.
    select(const primitive_id& id,
           const primitive_id& mask,
           const primitive_id& input,
           const primitive_id& input2,
           const padding& output_padding = padding(),
           const std::string& broadcast_type = "numpy")
        : primitive_base(id, {mask, input, input2}, output_padding),
          broadcast_type(broadcast_type) {}

    /// @brief String which determines broadcast type.
    std::string broadcast_type;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
