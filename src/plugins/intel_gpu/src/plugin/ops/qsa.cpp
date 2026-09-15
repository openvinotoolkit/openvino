// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/qsa.hpp"

// NOTE: unlike most other gpu_opset ops in this directory, QSA is lowered from the base
// ov::op::internal::QSA type (src/core/dev_api/openvino/op/qsa.hpp) rather than from
// ov::intel_gpu::op::QSA directly. ov::intel_gpu::op::QSA ("gpu_opset") publicly inherits
// from it and adds no new fields/behavior, so ProgramBuilder::CreateSingleLayerPrimitive's
// type_info->parent walk already dispatches gpu_opset QSA nodes to a factory registered
// for the base type - no separate exact-match registration (and no
// `using QSA = ov::intel_gpu::op::QSA;` alias hack, which would now conflict with the
// real ov::op::internal::QSA class) is needed. This lets plain REGISTER_FACTORY_IMPL
// keep working unmodified, exactly like every other op in this directory.

namespace ov::intel_gpu {

static void CreateQSAOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::QSA>& op) {
    validate_inputs_count(op, {ov::op::internal::QSA::InputIdx::COUNT});
    const auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);

    // compress_ratio == 0 marks a non-QSA (Gated DeltaNet) layer; such a layer must never
    // reach lowering as a QSA primitive.
    std::string reason;
    OPENVINO_ASSERT(ov::op::internal::QSA::validate_config(op->get_config(), reason), "[GPU] Cannot lower QSA op '", layer_name, "': ", reason);

    cldnn::qsa prim(layer_name, inputs, op->get_config());
    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, QSA);

}  // namespace ov::intel_gpu
