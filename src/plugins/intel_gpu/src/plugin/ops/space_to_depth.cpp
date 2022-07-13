// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/space_to_depth.hpp"

#include "intel_gpu/primitives/space_to_depth.hpp"

namespace ov {
namespace intel_gpu {

static cldnn::space_to_depth::depth_mode GetDepthMode(ngraph::op::v0::SpaceToDepth::SpaceToDepthMode mode) {
    switch (mode) {
        case ngraph::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST: return cldnn::space_to_depth::blocks_first;
        case ngraph::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST: return cldnn::space_to_depth::depth_first;
        default: IE_THROW() << "Unsupported SpaceToDepthMode value: " << static_cast<int>(mode);
    }
    return cldnn::space_to_depth::blocks_first;
}

static void CreateSpaceToDepthOp(Program& p, const std::shared_ptr<ngraph::op::v0::SpaceToDepth>& op) {
    p.ValidateInputs(op, {1});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);
    auto spaceToDepthPrim = cldnn::space_to_depth(layerName,
                                                  inputPrimitives[0],
                                                  GetDepthMode(op->get_mode()),
                                                  op->get_block_size(),
                                                  op->get_friendly_name());

    p.AddPrimitive(spaceToDepthPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, SpaceToDepth);

}  // namespace intel_gpu
}  // namespace ov
