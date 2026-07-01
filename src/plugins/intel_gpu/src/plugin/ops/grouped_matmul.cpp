// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/grouped_matmul.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "openvino/op/grouped_matmul.hpp"

namespace ov::intel_gpu {

static void CreateGroupedMatMulOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v17::GroupedMatMul>& op) {
    // GPU currently supports only the 2D x 3D case of GroupedMatMul-17
    // (mat_a: [total_tokens, K], mat_b: [G, N, K], offsets: [G]).
    validate_inputs_count(op, {3});
    auto inputs = p.GetInputInfo(op);

    const auto& a_shape = op->get_input_partial_shape(0);
    const auto& b_shape = op->get_input_partial_shape(1);
    OPENVINO_ASSERT(a_shape.rank().is_static() && a_shape.size() == 2,
                    "[GPU] GroupedMatMul: only 2D mat_a is supported, got shape ", a_shape);
    OPENVINO_ASSERT(b_shape.rank().is_static() && b_shape.size() == 3,
                    "[GPU] GroupedMatMul: only 3D mat_b is supported, got shape ", b_shape);

    // onednn grouped_gemm impl requires i32 offsets; insert a reorder if the graph feeds i64.
    const auto offsets_dtype = cldnn::element_type_to_data_type(op->get_input_element_type(2));
    if (offsets_dtype == cldnn::data_types::i64) {
        auto reorder_id = inputs[2].pid + "_" + op->get_friendly_name() + ProgramBuilder::m_preProcessTag;
        auto fmt = cldnn::format::get_default_format(op->get_input_partial_shape(2).size());
        auto reorder_prim = cldnn::reorder(reorder_id, inputs[2], fmt, cldnn::data_types::i32);
        p.add_primitive(*op, reorder_prim);
        inputs[2] = cldnn::input_info(reorder_id);
    }

    const std::string layer_name = layer_type_name_ID(op);
    const auto output_dt = cldnn::element_type_to_data_type(op->get_output_element_type(0));
    const cldnn::grouped_matmul prim(layer_name, inputs, output_dt);

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v17, GroupedMatMul);

}  // namespace ov::intel_gpu
