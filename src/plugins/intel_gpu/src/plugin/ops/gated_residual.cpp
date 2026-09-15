// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gated_residual.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/gated_residual.hpp"

// NOTE: unlike most other gpu_opset ops in this directory, GatedResidual is lowered from
// the base ov::op::internal::GatedResidual type
// (src/core/dev_api/openvino/op/gated_residual.hpp) rather than from
// ov::intel_gpu::op::GatedResidual directly. ov::intel_gpu::op::GatedResidual
// ("gpu_opset") publicly inherits from it and adds no new fields/behavior, so
// ProgramBuilder::CreateSingleLayerPrimitive's type_info->parent walk already dispatches
// gpu_opset GatedResidual nodes to a factory registered for the base type - no separate
// exact-match registration (and no
// `using GatedResidual = ov::intel_gpu::op::GatedResidual;` alias hack, which would now
// conflict with the real ov::op::internal::GatedResidual class) is needed. This lets
// plain REGISTER_FACTORY_IMPL keep working unmodified, exactly like every other op in
// this directory.

namespace ov::intel_gpu {

static void CreateGatedResidualOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::GatedResidual>& op) {
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
