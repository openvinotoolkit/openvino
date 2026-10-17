// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_sdpa.hpp"
#include "openvino/op/constant.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/sparse_sdpa.hpp"

// v1 scope: only the sel_indices/sel_count plan form is lowered; an ordinary (non-plan)
// attn_mask is not yet implemented in GPU lowering either (explicit rejection, not a
// silent dense/mask fallback -- see qsa_final_design_implement.md section 3.1/4.5), and a
// SCALE input must be a Constant so its value can be folded into the primitive's scale
// field (no non-constant scale support in this pass).

namespace ov::intel_gpu {

static void CreateSparseSDPAOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::SparseSDPA>& op) {
    using In = ov::op::internal::SparseSDPA::InputIdx;
    const auto layer_name = layer_type_name_ID(op);

    OPENVINO_ASSERT(!op->has_attn_mask(), "[GPU] SparseSDPA '", layer_name, "': attn_mask input is not supported by this lowering pass");

    std::string reason;
    OPENVINO_ASSERT(ov::op::internal::SparseSDPA::validate_config(op->get_config(), reason), "[GPU] Cannot lower SparseSDPA op '", layer_name, "': ", reason);
    OPENVINO_ASSERT(op->get_config().h_sel == 1, "[GPU] SparseSDPA '", layer_name, "': only h_sel == 1 is supported, got ", op->get_config().h_sel);
    for (size_t idx : {In::QUERY, In::KEY, In::VALUE}) {
        OPENVINO_ASSERT(op->get_input_element_type(idx) == ov::element::f16,
                        "[GPU] SparseSDPA '", layer_name, "': input ", idx, " must be f16, got ", op->get_input_element_type(idx));
    }

    float resolved_scale = 1.0f / std::sqrt(static_cast<float>(op->get_config().k_head_size));
    if (op->has_scale_input()) {
        auto scale_const = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(In::SCALE));
        OPENVINO_ASSERT(scale_const != nullptr, "[GPU] SparseSDPA '", layer_name, "': scale input must be a Constant in this lowering pass");
        resolved_scale = scale_const->cast_vector<float>().at(0);
    }

    auto full_inputs = p.GetInputInfo(op);
    const std::vector<cldnn::input_info> inputs(full_inputs.begin(), full_inputs.begin() + In::SEL_COUNT + 1);

    cldnn::sparse_sdpa prim(layer_name, inputs, op->get_config(), resolved_scale);
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, SparseSDPA);

}  // namespace ov::intel_gpu
