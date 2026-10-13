// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_pa.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/sparse_pa.hpp"

namespace ov::intel_gpu {

static void CreateSparsePAOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::SparsePA>& op) {
    validate_inputs_count(op, {ov::op::internal::SparsePA::InputIdx::COUNT});
    const auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);

    std::string reason;
    OPENVINO_ASSERT(ov::op::internal::SparsePA::validate_config(op->get_config(), reason), "[GPU] Cannot lower SparsePA op '", layer_name, "': ", reason);

    cldnn::sparse_pa prim(layer_name, inputs, op->get_config());
    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, SparsePA);

}  // namespace ov::intel_gpu
