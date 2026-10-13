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

    using In = ov::op::internal::SparsePA::InputIdx;
    OPENVINO_ASSERT(op->get_config().h_sel == 1, "[GPU] SparsePA '", layer_name, "': only h_sel == 1 is supported, got ", op->get_config().h_sel);
    for (size_t idx : {In::QUERY, In::KEY, In::VALUE, In::KEY_CACHE, In::VALUE_CACHE}) {
        OPENVINO_ASSERT(op->get_input_element_type(idx) == ov::element::f16,
                        "[GPU] SparsePA '", layer_name, "': input ", idx, " must be f16, got ", op->get_input_element_type(idx));
    }

    cldnn::sparse_pa prim(layer_name, inputs, op->get_config());
    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, SparsePA);

}  // namespace ov::intel_gpu
