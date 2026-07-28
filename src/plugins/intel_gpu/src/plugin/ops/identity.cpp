// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/identity.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/reorder.hpp"

namespace ov::intel_gpu {

static void CreateIdentityOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v16::Identity>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto reorderPrim = cldnn::reorder(layerName, inputs[0], cldnn::format::any, op->get_element_type());
    reorderPrim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, reorderPrim);
}

REGISTER_FACTORY_IMPL(v16, Identity);

}  // namespace ov::intel_gpu
