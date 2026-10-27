// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/sparse_sdpa.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/sparse_sdpa.hpp"
#include "openvino/op/constant.hpp"

// Only the selected_block_indices/selected_block_count plan form is lowered. A SCALE input must be a Constant so
// its value can be folded into the primitive's scale field, and attention sinks are not
// supported by the GPU kernels (explicit rejection, not a silent fallback).

namespace ov::op::internal {
using SparseSDPA = ov::intel_gpu::op::SparseSDPA;
}  // namespace ov::op::internal

namespace ov::intel_gpu {

static void CreateSparseSDPAOp(ProgramBuilder& p, const std::shared_ptr<ov::intel_gpu::op::SparseSDPA>& op) {
    using In = ov::intel_gpu::op::SparseSDPA::InputIdx;
    const auto layer_name = layer_type_name_ID(op);

    OPENVINO_ASSERT(!op->has_sinks(), "[GPU] SparseSDPA '", layer_name, "': the sinks input is not supported by the GPU implementation");

    std::string reason;
    OPENVINO_ASSERT(ov::intel_gpu::op::SparseSDPA::validate_config(op->get_config(), reason), "[GPU] Cannot lower SparseSDPA op '", layer_name, "': ", reason);
    OPENVINO_ASSERT(op->get_config().num_selection_heads == 1,
                    "[GPU] SparseSDPA '",
                    layer_name,
                    "': only num_selection_heads == 1 is supported, got ",
                    op->get_config().num_selection_heads);
    for (size_t idx : {In::QUERY, In::KEY, In::VALUE}) {
        OPENVINO_ASSERT(op->get_input_element_type(idx) == ov::element::f16,
                        "[GPU] SparseSDPA '",
                        layer_name,
                        "': input ",
                        idx,
                        " must be f16, got ",
                        op->get_input_element_type(idx));
    }

    float resolved_scale = 1.0f / std::sqrt(static_cast<float>(op->get_config().k_head_size));
    if (op->has_scale_input()) {
        auto scale_const = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(In::SCALE));
        OPENVINO_ASSERT(scale_const != nullptr, "[GPU] SparseSDPA '", layer_name, "': scale input must be a Constant in this lowering pass");
        resolved_scale = scale_const->cast_vector<float>().at(0);
    }

    auto full_inputs = p.GetInputInfo(op);
    const std::vector<cldnn::input_info> inputs(full_inputs.begin(), full_inputs.begin() + In::KEY_LENGTH + 1);

    cldnn::sparse_sdpa prim(layer_name, inputs, op->get_config(), resolved_scale);
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, SparseSDPA);

}  // namespace ov::intel_gpu
