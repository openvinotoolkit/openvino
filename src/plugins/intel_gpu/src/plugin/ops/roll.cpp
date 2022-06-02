// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/roll.hpp"

#include <ngraph/op/roll.hpp>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

namespace {

void CreateRollOp(Program& p, const std::shared_ptr<ngraph::op::v7::Roll>& op) {
    p.ValidateInputs(op, {3});

    const auto inputs = p.GetInputPrimitiveIDs(op);
    const auto layer_name = layer_type_name_ID(op);
    const auto& op_friendly_name = op->get_friendly_name();
    const auto& input_shape = op->get_input_shape(0);
    const uint8_t rank = input_shape.size();
    const auto format = DefaultFormatForDims(rank);
    const auto default_rank = format.dimension();

    auto shift_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));
    if (!shift_constant) {
        IE_THROW() << "Unsupported parameter node type in " << op_friendly_name << " (" << op->get_type_name() << ")";
    }
    const auto shift_raw = shift_constant->cast_vector<int32_t>();

    auto axes_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(2));
    if (!axes_constant) {
        IE_THROW() << "Unsupported parameter node type in " << op_friendly_name << " (" << op->get_type_name() << ")";
    }
    auto axes_raw = axes_constant->cast_vector<int32_t>();

    // Normalize axes and sum shift
    std::vector<int32_t> shift(default_rank);
    for (size_t a = 0; a < axes_raw.size(); ++a) {
        auto& axis = axes_raw[a];
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            IE_THROW() << op_friendly_name << " Incorrect axis value: " << axis;
        }
        shift[axis] += shift_raw[a];
    }

    // Normalize shift
    for (size_t s = 0; s < rank; ++s) {
        auto& sh = shift[s];
        const auto dim = static_cast<int32_t>(input_shape[s]);
        sh %= dim;
        if (sh < 0) {
            sh += dim;
        }
    }

    const cldnn::roll roll_prim(layer_name, inputs.front(), {format, shift}, op_friendly_name);
    p.AddPrimitive(roll_prim);
    p.AddPrimitiveToProfiler(op);
}

}  // namespace

REGISTER_FACTORY_IMPL(v7, Roll);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
