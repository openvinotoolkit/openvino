// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/space_to_depth.hpp"

#include "intel_gpu/primitives/space_to_depth.hpp"

namespace ov {
namespace intel_gpu {

static cldnn::space_to_depth::depth_mode GetDepthMode(ov::op::v0::SpaceToDepth::SpaceToDepthMode mode) {
    switch (mode) {
        case ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST: return cldnn::space_to_depth::blocks_first;
        case ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST: return cldnn::space_to_depth::depth_first;
        default: OPENVINO_THROW("[GPU] Unsupported SpaceToDepthMode value: ", static_cast<int>(mode));
    }
    return cldnn::space_to_depth::blocks_first;
}

static void CreateSpaceToDepthOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::SpaceToDepth>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);
    auto spaceToDepthPrim = cldnn::space_to_depth(layerName,
                                                  inputs[0],
                                                  GetDepthMode(op->get_mode()),
                                                  op->get_block_size());

    p.add_primitive(*op, spaceToDepthPrim);
}

REGISTER_FACTORY_IMPL(v0, SpaceToDepth);

}  // namespace intel_gpu
}  // namespace ov
