// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"

#include "openvino/op/select.hpp"

#include "intel_gpu/primitives/select.hpp"

namespace ov {
namespace intel_gpu {

static void CreateSelectOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Select>& op) {
    validate_inputs_count(op, {3});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto broadcast_type = op->get_auto_broadcast();

    if (broadcast_type.m_type != ov::op::AutoBroadcastType::NONE &&
        broadcast_type.m_type != ov::op::AutoBroadcastType::NUMPY) {
        OPENVINO_THROW("[GPU] Unsupported broadcast type (", broadcast_type.m_type, ") in layer " + op->get_friendly_name());
    }

    auto selectPrim = cldnn::select(layerName,
                                    inputs[0],
                                    inputs[1],
                                    inputs[2],
                                    broadcast_type);

    p.add_primitive(*op, selectPrim);
}

REGISTER_FACTORY_IMPL(v1, Select);

}  // namespace intel_gpu
}  // namespace ov
