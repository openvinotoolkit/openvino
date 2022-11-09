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
    /// @param spec Auto broadcast rule specification
    /// @param output_padding Output data padding information.
    /// "numpy" means that numpy-tyle (ONNX) broadcasting is allowed,
    /// "none" means that all inputs need to have the same shape.
    select(const primitive_id& id,
           const primitive_id& mask,
           const primitive_id& input,
           const primitive_id& input2,
           const ov::op::AutoBroadcastSpec& spec = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY),
           const padding& output_padding = padding())
        : primitive_base(id, {mask, input, input2}, output_padding),
          broadcast_spec(spec.m_type, spec.m_axis) {}

    /// @brief Define auto broadcast rule specification.
    ov::op::AutoBroadcastSpec broadcast_spec;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
