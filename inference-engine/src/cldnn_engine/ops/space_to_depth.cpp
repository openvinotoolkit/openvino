// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/space_to_depth.hpp"

#include "api/space_to_depth.hpp"

namespace CLDNNPlugin {

static cldnn::space_to_depth::depth_mode GetDepthMode(ngraph::op::v0::SpaceToDepth::SpaceToDepthMode mode) {
    switch (mode) {
        case ngraph::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST: return cldnn::space_to_depth::blocks_first;
        case ngraph::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST: return cldnn::space_to_depth::depth_first;
        default: THROW_IE_EXCEPTION << "Unsupported SpaceToDepthMode value: " << static_cast<int>(mode);
    }
    return cldnn::space_to_depth::blocks_first;
}

void CreateSpaceToDepthOp(Program& p, const std::shared_ptr<ngraph::op::v0::SpaceToDepth>& op) {
    p.ValidateInputs(op, {1});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);
    auto spaceToDepthPrim = cldnn::space_to_depth(layerName,
                                                  inputPrimitives[0],
                                                  GetDepthMode(op->get_mode()),
                                                  op->get_block_size());

    p.AddPrimitive(spaceToDepthPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, SpaceToDepth);

}  // namespace CLDNNPlugin
