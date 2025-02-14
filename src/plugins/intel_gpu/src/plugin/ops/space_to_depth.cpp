// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/space_to_depth.hpp"

#include "intel_gpu/primitives/space_to_depth.hpp"

namespace ov::intel_gpu {
static void CreateSpaceToDepthOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::SpaceToDepth>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);
    auto spaceToDepthPrim = cldnn::space_to_depth(layerName,
                                                  inputs[0],
                                                  op->get_mode(),
                                                  op->get_block_size());

    p.add_primitive(*op, spaceToDepthPrim);
}

REGISTER_FACTORY_IMPL(v0, SpaceToDepth);

}  // namespace ov::intel_gpu
