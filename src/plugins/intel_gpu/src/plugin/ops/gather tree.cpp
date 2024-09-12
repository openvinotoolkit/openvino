// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/gather_tree.hpp"

#include "intel_gpu/primitives/gather_tree.hpp"
#include "intel_gpu/primitives/reorder.hpp"

namespace ov {
namespace intel_gpu {

static void CreateGatherTreeOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::GatherTree>& op) {
    validate_inputs_count(op, {4});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto gatherTreePrim = cldnn::gather_tree(layerName,
                                             inputs[0],
                                             inputs[1],
                                             inputs[2],
                                             inputs[3]);

    p.add_primitive(*op, gatherTreePrim);
}

REGISTER_FACTORY_IMPL(v1, GatherTree);

}  // namespace intel_gpu
}  // namespace ov
