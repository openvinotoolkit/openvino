// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/scatter_update.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/primitives/scatter_update.hpp"

namespace ov::intel_gpu {

static void CreateScatterUpdateOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v3::ScatterUpdate>& op) {
    validate_inputs_count(op, {4});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto axes_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(3));
    OPENVINO_ASSERT(axes_constant != nullptr, "[GPU] Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
    int64_t axis = axes_constant->cast_vector<int64_t>()[0];
    auto primitive = cldnn::scatter_update(layerName,
                                           inputs[0],
                                           inputs[1],
                                           inputs[2],
                                           axis);

    p.add_primitive(*op, primitive);
}

REGISTER_FACTORY_IMPL(v3, ScatterUpdate);

}  // namespace ov::intel_gpu
