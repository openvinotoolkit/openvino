// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/sparse_pa.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/sparse_pa.hpp"
#include "openvino/op/constant.hpp"

namespace ov::op::internal {
using SparsePA = ov::intel_gpu::op::SparsePA;
}  // namespace ov::op::internal

namespace ov::intel_gpu {

static void CreateSparsePAOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::SparsePA>& op) {
    using In = ov::intel_gpu::op::SparsePA::InputIdx;

    validate_inputs_count(op, {In::COUNT});
    const auto layer_name = layer_type_name_ID(op);

    std::string reason;
    OPENVINO_ASSERT(ov::intel_gpu::op::SparsePA::validate_config(op->get_config(), reason), "[GPU] Cannot lower SparsePA op '", layer_name, "': ", reason);
    OPENVINO_ASSERT(op->get_config().num_selection_heads == 1,
                    "[GPU] SparsePA '",
                    layer_name,
                    "': only num_selection_heads == 1 is supported, got ",
                    op->get_config().num_selection_heads);
    for (size_t idx : {In::QUERY, In::KEY, In::VALUE, In::KEY_CACHE, In::VALUE_CACHE}) {
        OPENVINO_ASSERT(op->get_input_element_type(idx) == ov::element::f16,
                        "[GPU] SparsePA '",
                        layer_name,
                        "': input ",
                        idx,
                        " must be f16, got ",
                        op->get_input_element_type(idx));
    }

    // Attention sinks are not supported by the GPU kernels (explicit rejection, not a silent fallback).
    OPENVINO_ASSERT(!op->has_sinks(), "[GPU] SparsePA '", layer_name, "': the sinks input is not supported by the GPU implementation");

    // The scale input is folded into the primitive, so it must be a Constant.
    auto scale_const = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(In::SCALE));
    OPENVINO_ASSERT(scale_const != nullptr, "[GPU] SparsePA '", layer_name, "': scale input must be a Constant in this lowering pass");
    const float scale = scale_const->cast_vector<float>().at(0);

    // Only QUERY..SELECTED_BLOCK_COUNT are consumed by the kernels; scale and sinks are not primitive inputs.
    const auto full_inputs = p.GetInputInfo(op);
    const std::vector<cldnn::input_info> inputs(full_inputs.begin(), full_inputs.begin() + cldnn::sparse_pa::InputIdx::COUNT);

    cldnn::sparse_pa prim(layer_name, inputs, op->get_config(), scale);
    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, SparsePA);

}  // namespace ov::intel_gpu
