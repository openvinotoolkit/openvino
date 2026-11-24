// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/space_to_depth.hpp"

#include "intel_gpu/op/grouped_space_to_depth.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/space_to_depth.hpp"

namespace ov::op::internal {
using GroupedSpaceToDepth = ov::intel_gpu::op::GroupedSpaceToDepth;
}

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

static void CreateGroupedSpaceToDepthOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::GroupedSpaceToDepth>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    auto primitive = cldnn::space_to_depth(layer_type_name_ID(op), inputs[0], op->get_factor_t(), op->get_factor_s(), op->get_output_channels());
    primitive.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, primitive);
}

REGISTER_FACTORY_IMPL(internal, GroupedSpaceToDepth);

}  // namespace ov::intel_gpu
