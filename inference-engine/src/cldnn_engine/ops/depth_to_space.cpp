// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/depth_to_space.hpp"

#include "api/depth_to_space.hpp"

namespace CLDNNPlugin {

static cldnn::depth_to_space_mode GetDepthMode(ngraph::op::v0::DepthToSpace::DepthToSpaceMode mode) {
    switch (mode) {
        case ngraph::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST:
            return cldnn::depth_to_space_mode::blocks_first;
        case ngraph::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST:
            return cldnn::depth_to_space_mode::depth_first;
        default: THROW_IE_EXCEPTION << "Unsupported DepthToSpaceMode value: " << static_cast<int>(mode);
    }
    return cldnn::depth_to_space_mode::blocks_first;
}

void CreateDepthToSpaceOp(Program& p, const std::shared_ptr<ngraph::op::v0::DepthToSpace>& op) {
    p.ValidateInputs(op, {1});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    size_t blockSize = op->get_block_size();
    cldnn::depth_to_space_mode mode = GetDepthMode(op->get_mode());

    auto depthToSpacePrim = cldnn::depth_to_space(layerName,
                                                  inputPrimitives[0],
                                                  blockSize,
                                                  mode);

    p.AddPrimitive(depthToSpacePrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, DepthToSpace);

}  // namespace CLDNNPlugin
