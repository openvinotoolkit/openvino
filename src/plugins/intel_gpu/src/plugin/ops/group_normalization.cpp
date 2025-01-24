// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_normalization.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/group_normalization.hpp"

namespace ov::intel_gpu {

static void CreateGroupNormalizationOp(ProgramBuilder& p, const std::shared_ptr<op::v12::GroupNormalization>& op) {
    validate_inputs_count(op, {3});
    auto inputs = p.GetInputInfo(op);
    auto layerName = layer_type_name_ID(op);
    cldnn::group_normalization groupNormalizationPrimitive {
        layerName,
        inputs[0],
        inputs[1],
        inputs[2],
        op->get_num_groups(),
        op->get_epsilon()
    };
    p.add_primitive(*op, groupNormalizationPrimitive);
}

REGISTER_FACTORY_IMPL(v12, GroupNormalization);

}  // namespace ov::intel_gpu
