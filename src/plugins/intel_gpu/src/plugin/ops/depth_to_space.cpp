// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/depth_to_space.hpp"

#include "intel_gpu/primitives/depth_to_space.hpp"

namespace ov {
namespace intel_gpu {

static cldnn::depth_to_space_mode GetDepthMode(ov::op::v0::DepthToSpace::DepthToSpaceMode mode) {
    switch (mode) {
        case ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST:
            return cldnn::depth_to_space_mode::blocks_first;
        case ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST:
            return cldnn::depth_to_space_mode::depth_first;
        default: OPENVINO_THROW("Unsupported DepthToSpaceMode value: ", static_cast<int>(mode));
    }
    return cldnn::depth_to_space_mode::blocks_first;
}

static void CreateDepthToSpaceOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::DepthToSpace>& op) {
    validate_inputs_count(op, {1});
    auto inputPrimitives = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    size_t blockSize = op->get_block_size();
    cldnn::depth_to_space_mode mode = GetDepthMode(op->get_mode());

    auto depthToSpacePrim = cldnn::depth_to_space(layerName,
                                                  inputPrimitives[0],
                                                  blockSize,
                                                  mode);

    p.add_primitive(*op, depthToSpacePrim);
}

REGISTER_FACTORY_IMPL(v0, DepthToSpace);

}  // namespace intel_gpu
}  // namespace ov
