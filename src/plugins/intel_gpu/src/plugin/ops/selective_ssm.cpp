// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/selective_ssm.hpp"

#include <intel_gpu/primitives/selective_ssm.hpp>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"

namespace ov::intel_gpu {

static void CreateSelectiveSSMOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::SelectiveSSM>& op) {
    validate_inputs_count(op, {6});
    auto inputs = p.GetInputInfo(op);

    const std::string layer_name = layer_type_name_ID(op);
    cldnn::selective_ssm prim(layer_name, inputs);
    prim.num_outputs = op->get_output_size();
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, SelectiveSSM);

}  // namespace ov::intel_gpu
