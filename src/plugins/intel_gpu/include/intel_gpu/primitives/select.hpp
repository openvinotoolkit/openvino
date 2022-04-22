// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
           const input_info& mask,
           const input_info& input,
           const input_info& input2,
           const primitive_id& ext_prim_id = "",
           const padding& output_padding = padding(),
           const std::string& broadcast_type = "numpy")
        : primitive_base(id, {mask, input, input2}, ext_prim_id, {output_padding}),
          broadcast_type(broadcast_type) {}

    /// @brief String which determines broadcast type.
    std::string broadcast_type;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
