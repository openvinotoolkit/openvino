// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

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

    select() : primitive_base("", {}) {}

    /// @brief Constructs select primitive.
    /// @param id This primitive id.
    /// @param mask Input primitive id with values needed for select computation.
    /// @param input Input primitive id.
    /// @param input2 Second input primitive id.
    /// @param spec Auto broadcast rule specification
    /// "numpy" means that numpy-tyle (ONNX) broadcasting is allowed,
    /// "none" means that all inputs need to have the same shape.
    select(const primitive_id& id,
           const input_info& mask,
           const input_info& input,
           const input_info& input2,
           const ov::op::AutoBroadcastSpec& spec = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY))
        : primitive_base(id, {mask, input, input2}),
          broadcast_spec(spec.m_type, spec.m_axis) {}

    /// @brief Define auto broadcast rule specification.
    ov::op::AutoBroadcastSpec broadcast_spec;

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const select>(rhs);

        return broadcast_spec == rhs_casted.broadcast_spec;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<select>::save(ob);
        ob << make_data(&broadcast_spec, sizeof(ov::op::AutoBroadcastSpec));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<select>::load(ib);
        ib >> make_data(&broadcast_spec, sizeof(ov::op::AutoBroadcastSpec));
    }
};
}  // namespace cldnn
