// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/except.hpp"
#include "intel_gpu/plugin/program_builder.hpp"

#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/primitives/broadcast.hpp"

namespace ov {
namespace intel_gpu {

static void CreateCommonBroadcastOp(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op, const ov::AxisSet axis_mapping) {
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto input_pshape = op->get_input_partial_shape(0);
    auto output_pshape = op->get_output_partial_shape(0);

    auto input = inputs[0];

    ov::op::BroadcastModeSpec mode = ov::op::BroadcastType::NONE;
    if (auto broadcast_v3 = std::dynamic_pointer_cast<ov::op::v3::Broadcast>(op)) {
        mode = broadcast_v3->get_broadcast_spec();
    } else if (auto broadcast_v1 = std::dynamic_pointer_cast<ov::op::v1::Broadcast>(op)) {
        switch (broadcast_v1->get_broadcast_spec().m_type) {
            case ov::op::AutoBroadcastType::NONE: mode = ov::op::BroadcastType::NONE; break;
            case ov::op::AutoBroadcastType::NUMPY: mode = ov::op::BroadcastType::NUMPY; break;
            case ov::op::AutoBroadcastType::PDPD: mode = ov::op::BroadcastType::PDPD; break;
            default:
                OPENVINO_THROW("[GPU] Can't match Broadcast v1 mode with v3 version");
        }
    } else {
        OPENVINO_THROW("[GPU] Can't cast Broadcast operation to any supported version");
    }

    std::shared_ptr<cldnn::broadcast> broadcast_prim = nullptr;
    if (output_pshape.is_static()) {
        broadcast_prim = std::make_shared<cldnn::broadcast>(layerName,
                                                            input,
                                                            output_pshape.to_shape(),
                                                            axis_mapping,
                                                            mode);
    } else {
        broadcast_prim = std::make_shared<cldnn::broadcast>(layerName,
                                                            input,
                                                            inputs[1],
                                                            axis_mapping,
                                                            mode);
    }

    broadcast_prim->output_pshape = op->get_output_partial_shape(0);

    p.add_primitive(*op, broadcast_prim);
}

static void CreateBroadcastOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Broadcast>& op) {
    validate_inputs_count(op, {2, 3});
    if (op->get_broadcast_spec().m_type == ov::op::AutoBroadcastType::NONE && op->get_input_size() == 3) {
        auto axis_mapping_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
        OPENVINO_ASSERT(axis_mapping_node != nullptr, "[GPU] Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");

        auto axis_mapping = axis_mapping_node->get_axis_set_val();
        CreateCommonBroadcastOp(p, op, axis_mapping);
    } else {
        // TODO: check if axis_mapping is not needed in these cases and prepending input shape with ones works fine in all cases
        CreateCommonBroadcastOp(p, op, {});
    }
}

static void CreateBroadcastOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v3::Broadcast>& op) {
    validate_inputs_count(op, {2, 3});
    ov::AxisSet axis_mapping;
    if (op->get_input_size() == 3) {
        auto axis_mapping_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
        OPENVINO_ASSERT(axis_mapping_node != nullptr, "[GPU] Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");

        axis_mapping = axis_mapping_node->get_axis_set_val();
    }
    CreateCommonBroadcastOp(p, op, axis_mapping);
}

REGISTER_FACTORY_IMPL(v1, Broadcast);
REGISTER_FACTORY_IMPL(v3, Broadcast);

}  // namespace intel_gpu
}  // namespace ov
