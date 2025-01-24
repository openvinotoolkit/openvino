// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/primitives/scatter_nd_update.hpp"

namespace ov::intel_gpu {

static void CreateScatterNDUpdateOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v3::ScatterNDUpdate>& op) {
    validate_inputs_count(op, {3});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);
    auto indices_rank = op->get_input_partial_shape(1).size();

    auto primitive = cldnn::scatter_nd_update(layerName,
                                              inputs[0],
                                              inputs[1],
                                              inputs[2],
                                              indices_rank);

    p.add_primitive(*op, primitive);
}

REGISTER_FACTORY_IMPL(v3, ScatterNDUpdate);

}  // namespace ov::intel_gpu
