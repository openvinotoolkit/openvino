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

    const auto x_ps = op->get_input_partial_shape(3);
    const auto B_ps = op->get_input_partial_shape(2);
    if (x_ps.rank().is_static() && B_ps.rank().is_static()) {
        const auto x_rank = x_ps.rank().get_length();
        const auto b_rank = B_ps.rank().get_length();
        if (x_rank >= 4 && b_rank >= 4) {
            if (x_ps[2].is_static())
                prim.num_heads = x_ps[2].get_length();
            if (x_ps[3].is_static())
                prim.head_dim = x_ps[3].get_length();
            if (B_ps[2].is_static())
                prim.num_groups = B_ps[2].get_length();
            if (B_ps[3].is_static())
                prim.state_size = B_ps[3].get_length();
        }
    }

    prim.num_outputs = op->get_output_size();
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, SelectiveSSM);

}  // namespace ov::intel_gpu
