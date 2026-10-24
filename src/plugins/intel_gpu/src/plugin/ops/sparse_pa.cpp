// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_pa.hpp"
#include "openvino/op/constant.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/sparse_pa.hpp"

namespace ov::intel_gpu {

static void CreateSparsePAOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::SparsePA>& op) {
    validate_inputs_count(op,
                          {ov::op::internal::SparsePA::InputIdx::COUNT_MIN,
                           ov::op::internal::SparsePA::InputIdx::COUNT_MIN + 1,
                           ov::op::internal::SparsePA::InputIdx::COUNT_MAX});
    const auto layer_name = layer_type_name_ID(op);

    std::string reason;
    OPENVINO_ASSERT(ov::op::internal::SparsePA::validate_config(op->get_config(), reason), "[GPU] Cannot lower SparsePA op '", layer_name, "': ", reason);

    using In = ov::op::internal::SparsePA::InputIdx;
    OPENVINO_ASSERT(op->get_config().num_selection_heads == 1, "[GPU] SparsePA '", layer_name, "': only num_selection_heads == 1 is supported, got ", op->get_config().num_selection_heads);
    for (size_t idx : {In::QUERY, In::KEY, In::VALUE, In::KEY_CACHE, In::VALUE_CACHE}) {
        OPENVINO_ASSERT(op->get_input_element_type(idx) == ov::element::f16,
                        "[GPU] SparsePA '", layer_name, "': input ", idx, " must be f16, got ", op->get_input_element_type(idx));
    }

    // Attention sinks are not supported by the GPU kernels (explicit rejection, not a silent fallback).
    OPENVINO_ASSERT(!op->has_sinks(), "[GPU] SparsePA '", layer_name, "': the sinks input is not supported by the GPU implementation");

    // Only the selected_block_indices/selected_block_count plan form is implemented.
    OPENVINO_ASSERT(!op->has_selected_block_mask(), "[GPU] SparsePA '", layer_name, "': the selected_block_mask input is not supported by the GPU implementation");

    // The scale input is folded into the primitive, so it must be a Constant.
    auto scale_const = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(In::SCALE));
    OPENVINO_ASSERT(scale_const != nullptr, "[GPU] SparsePA '", layer_name, "': scale input must be a Constant in this lowering pass");
    const float scale = scale_const->cast_vector<float>().at(0);

    // Only QUERY..SELECTED_BLOCK_COUNT are consumed by the kernels; scale, sinks and qb_start are not primitive inputs.
    const auto full_inputs = p.GetInputInfo(op);
    const std::vector<cldnn::input_info> inputs(full_inputs.begin(), full_inputs.begin() + In::SELECTED_BLOCK_COUNT + 1);

    cldnn::sparse_pa prim(layer_name, inputs, op->get_config(), scale);
    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, SparsePA);

}  // namespace ov::intel_gpu
