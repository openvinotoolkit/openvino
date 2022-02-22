// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/depth_to_space.hpp"

#include "intel_gpu/primitives/depth_to_space.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

static cldnn::depth_to_space_mode GetDepthMode(ngraph::op::v0::DepthToSpace::DepthToSpaceMode mode) {
    switch (mode) {
        case ngraph::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST:
            return cldnn::depth_to_space_mode::blocks_first;
        case ngraph::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST:
            return cldnn::depth_to_space_mode::depth_first;
        default: IE_THROW() << "Unsupported DepthToSpaceMode value: " << static_cast<int>(mode);
    }
    return cldnn::depth_to_space_mode::blocks_first;
}

static void CreateDepthToSpaceOp(Program& p, const std::shared_ptr<ngraph::op::v0::DepthToSpace>& op) {
    p.ValidateInputs(op, {1});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    size_t blockSize = op->get_block_size();
    cldnn::depth_to_space_mode mode = GetDepthMode(op->get_mode());

    auto depthToSpacePrim = cldnn::depth_to_space(layerName,
                                                  inputPrimitives[0],
                                                  blockSize,
                                                  mode,
                                                  op->get_friendly_name());

    p.AddPrimitive(depthToSpacePrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, DepthToSpace);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
