// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/qsa.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/qsa.hpp"

namespace ov {
namespace op {
namespace internal {
using QSA = ov::intel_gpu::op::QSA;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreateQSAOp(ProgramBuilder& p, const std::shared_ptr<ov::intel_gpu::op::QSA>& op) {
    validate_inputs_count(op, {ov::intel_gpu::op::QSA::InputIdx::COUNT});
    const auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);

    // compress_ratio == 0 marks a non-QSA (Gated DeltaNet) layer; such a layer must never
    // reach lowering as a QSA primitive.
    std::string reason;
    OPENVINO_ASSERT(ov::intel_gpu::op::QSA::validate_config(op->get_config(), reason), "[GPU] Cannot lower QSA op '", layer_name, "': ", reason);

    cldnn::qsa prim(layer_name, inputs, op->get_config());
    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, QSA);

}  // namespace ov::intel_gpu
