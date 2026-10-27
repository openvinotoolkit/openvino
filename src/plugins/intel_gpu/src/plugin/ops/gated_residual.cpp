// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/gated_residual.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/gated_residual.hpp"

namespace ov::op::internal {
using GatedResidual = ov::intel_gpu::op::GatedResidual;
}  // namespace ov::op::internal

namespace ov::intel_gpu {

static void CreateGatedResidualOp(ProgramBuilder& p, const std::shared_ptr<ov::intel_gpu::op::GatedResidual>& op) {
    validate_inputs_count(op, {3, 4, 5});
    const auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);

    cldnn::gated_residual prim(layer_name,
                               inputs,
                               op->get_mode(),
                               op->get_hc_count(),
                               op->get_hidden_size(),
                               op->get_lowrank(),
                               op->get_epsilon(),
                               op->get_norm_gain_plus_one(),
                               op->get_produce_write_gate());

    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, GatedResidual);

}  // namespace ov::intel_gpu
