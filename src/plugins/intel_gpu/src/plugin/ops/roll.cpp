// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roll.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/primitives/roll.hpp"
#include "intel_gpu/plugin/program_builder.hpp"

namespace ov {
namespace intel_gpu {

namespace {

void CreateRollOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v7::Roll>& op) {
    validate_inputs_count(op, {3});

    const auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);
    const auto& op_friendly_name = op->get_friendly_name();

    auto shift_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    OPENVINO_ASSERT(shift_constant != nullptr, "[GPU] Unsupported parameter nodes type in ", op_friendly_name, " (", op->get_type_name(), ")");
    const auto shift_raw = shift_constant->cast_vector<int32_t>();

    auto axes_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
    OPENVINO_ASSERT(axes_constant != nullptr, "[GPU] Unsupported parameter nodes type in ", op_friendly_name, " (", op->get_type_name(), ")");
    auto axes_raw = axes_constant->cast_vector<int32_t>();

    const cldnn::roll roll_prim(layer_name, inputs.front(), shift_raw, axes_raw);
    p.add_primitive(*op, roll_prim);
}

}  // namespace

REGISTER_FACTORY_IMPL(v7, Roll);

}  // namespace intel_gpu
}  // namespace ov
