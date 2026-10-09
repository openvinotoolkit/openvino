// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/depth_to_space.hpp"

#include "intel_gpu/primitives/depth_to_space.hpp"
#include "intel_gpu/op/grouped_depth_to_space.hpp"

namespace ov::op::internal {
using GroupedDepthToSpace = ov::intel_gpu::op::GroupedDepthToSpace;
}

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

static void CreateGroupedDepthToSpaceOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::GroupedDepthToSpace>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    auto prim =
        cldnn::depth_to_space(layer_type_name_ID(op), inputs[0], op->get_factor_t(), op->get_factor_s(), op->get_output_channels(), op->get_crop_begin_t());
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, GroupedDepthToSpace);

}  // namespace intel_gpu
}  // namespace ov
