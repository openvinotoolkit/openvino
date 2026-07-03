// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/grouped_matmul.hpp"

#include "intel_gpu/op/grouped_matmul_compressed.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/grouped_matmul.hpp"

namespace ov {
namespace op {
namespace internal {
using GroupedMatMulCompressed = ov::intel_gpu::op::GroupedMatMulCompressed;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

// onednn grouped_gemm impl requires i32 offsets; insert a reorder if the graph feeds i64.
static void ensure_i32_offsets(ProgramBuilder& p,
                               const std::shared_ptr<ov::Node>& op,
                               std::vector<cldnn::input_info>& inputs,
                               size_t offsets_idx) {
    const auto offsets_dtype = cldnn::element_type_to_data_type(op->get_input_element_type(offsets_idx));
    if (offsets_dtype == cldnn::data_types::i32)
        return;
    auto reorder_id = inputs[offsets_idx].pid + "_" + op->get_friendly_name() + ProgramBuilder::m_preProcessTag;
    auto fmt = cldnn::format::get_default_format(op->get_input_partial_shape(offsets_idx).size());
    auto reorder_prim = cldnn::reorder(reorder_id, inputs[offsets_idx], fmt, cldnn::data_types::i32);
    p.add_primitive(*op, reorder_prim);
    inputs[offsets_idx] = cldnn::input_info(reorder_id);
}

static void CreateGroupedMatMulOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v17::GroupedMatMul>& op) {
    // GroupedMatMul-17 has two cases:
    //   2D x 3D: mat_a[T,K] x mat_b[G,N,K] + offsets[G] -> [T,N]  (MoE forward pass)
    //   3D x 3D: mat_a[G,M,K] x mat_b[G,N,K]            -> [G,M,N] (batched uniform)
    //
    // The 3D x 3D case is a plain batched matmul with transpose_b=true (since mat_b is
    // stored pre-transposed as [G, N, K]), so we lower it here to cldnn::fully_connected
    // (with weights_rank=3) and let the regular FC impls handle it. The 2D x 3D case has
    // no direct equivalent and is dispatched to the specialized cldnn::grouped_matmul
    // primitive.
    validate_inputs_count(op, {2, 3});
    auto inputs = p.GetInputInfo(op);

    const auto& a_shape = op->get_input_partial_shape(0);
    const auto& b_shape = op->get_input_partial_shape(1);
    OPENVINO_ASSERT(a_shape.rank().is_static() && (a_shape.size() == 2 || a_shape.size() == 3),
                    "[GPU] GroupedMatMul: mat_a must be rank 2 or 3, got shape ", a_shape);
    OPENVINO_ASSERT(b_shape.rank().is_static() && b_shape.size() == 3,
                    "[GPU] GroupedMatMul: only 3D mat_b is supported, got shape ", b_shape);

    const std::string layer_name = layer_type_name_ID(op);
    const auto output_dt = cldnn::element_type_to_data_type(op->get_output_element_type(0));
    if (a_shape.size() == 3) {
        // 3D x 3D: lower to a batched fully_connected with transpose_b=true.
        //   A:[G,M,K] x B:[G,N,K]  ==  A @ B.T  ->  [G,M,N]
        auto fc_prim = cldnn::fully_connected(layer_name,
                                              inputs[0],
                                              /*weights=*/inputs[1].pid,
                                              /*bias=*/"",
                                              output_dt,
                                              /*input_size=*/3,
                                              /*weights_rank=*/3,
                                              /*weights_transposed=*/true);
        p.add_primitive(*op, fc_prim);
        return;
    }

    // 2D x 3D: dispatch to the specialized grouped_matmul primitive.
    OPENVINO_ASSERT(inputs.size() == 3,
                    "[GPU] GroupedMatMul: 2D x 3D case requires 3 inputs (mat_a, mat_b, offsets)");

    ensure_i32_offsets(p, op, inputs, /*offsets_idx=*/2);

    const cldnn::grouped_matmul prim(layer_name, inputs, output_dt);
    p.add_primitive(*op, prim);
}

static void CreateGroupedMatMulCompressedOp(ProgramBuilder& p,
                                            const std::shared_ptr<op::GroupedMatMulCompressed>& op) {
    // Two layouts, distinguished by the presence of the offsets input:
    //   2D x 3D: [mat_a, mat_b(compressed), offsets, scale, (zp)]
    //            -> cldnn::grouped_matmul (specialized primitive)
    //   3D x 3D: [mat_a, mat_b(compressed), scale, (zp)]
    //            -> cldnn::fully_connected (batched, weights_rank=3, compressed)
    validate_inputs_count(op, {3, 4, 5});
    auto inputs = p.GetInputInfo(op);

    const auto& a_shape = op->get_input_partial_shape(0);
    const auto& b_shape = op->get_input_partial_shape(1);
    OPENVINO_ASSERT(a_shape.rank().is_static() && (a_shape.size() == 2 || a_shape.size() == 3),
                    "[GPU] GroupedMatMulCompressed: mat_a must be rank 2 or 3, got shape ", a_shape);
    OPENVINO_ASSERT(b_shape.rank().is_static() && b_shape.size() == 3,
                    "[GPU] GroupedMatMulCompressed: mat_b must be rank 3, got shape ", b_shape);

    const std::string layer_name = layer_type_name_ID(op);
    const auto output_dt = cldnn::element_type_to_data_type(op->get_output_element_type(0));
    const bool with_offsets = op->has_offsets();

    if (!with_offsets) {
        // 3D x 3D compressed path: lower to a batched compressed fully_connected.
        //   A:[G,M,K] x B:[G,N,K]  ==  A @ B.T  ->  [G,M,N] with weights_transposed=true.
        OPENVINO_ASSERT(inputs.size() >= 3, "[GPU] GroupedMatMulCompressed 3Dx3D requires at least 3 inputs (mat_a, mat_b, scale)");

        const auto weights_pid = inputs[1].pid;
        const auto scale_pid = inputs[2].pid;

        cldnn::primitive_id zp_pid = "";
        float zp_scalar_value = 0.0f;
        bool has_scalar_zp = false;
        if (op->get_input_size() > 3) {
            constexpr size_t zp_idx = 3;
            zp_pid = inputs[zp_idx].pid;
            auto zp_const = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(zp_idx));
            if (zp_const && ov::shape_size(zp_const->get_output_shape(0)) == 1) {
                has_scalar_zp = true;
                zp_scalar_value = zp_const->cast_vector<float>()[0];
            }
        }

        auto fc_prim = cldnn::fully_connected(layer_name,
                                              inputs[0],
                                              weights_pid /*weights=*/,
                                              "" /*bias=*/,
                                              scale_pid /*decompression_scale=*/,
                                              zp_pid /*decompression_zero_point=*/,
                                              output_dt,
                                              3 /*input_size=*/,
                                              3 /*weights_rank=*/,
                                              true /*weights_transposed=*/);
        if (has_scalar_zp) {
            fc_prim.decompression_zero_point_scalar = zp_scalar_value;
        }
        p.add_primitive(*op, fc_prim);

        return;
    }

    // 2D x 3D compressed path.
    OPENVINO_ASSERT(inputs.size() >= 4, "[GPU] GroupedMatMulCompressed 2Dx3D requires at least 4 inputs (mat_a, mat_b, offsets, scale)");

    ensure_i32_offsets(p, op, inputs, /*offsets_idx=*/2);

    const auto scale = inputs[3];
    cldnn::input_info zp = {};
    float zp_scalar_value = 0.0f;
    bool has_scalar_zp = false;
    if (op->get_input_size() > 4) {
        constexpr size_t zp_idx = 4;
        zp = inputs[zp_idx];
        auto zp_const = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(zp_idx));
        if (zp_const && ov::shape_size(zp_const->get_output_shape(0)) == 1) {
            has_scalar_zp = true;
            zp_scalar_value = zp_const->cast_vector<float>()[0];
        }
    }

    // base inputs are mat_a / mat_b / offsets; scale + zp are attached via primitive fields.
    std::vector<cldnn::input_info> base_inputs{inputs[0], inputs[1], inputs[2]};
    cldnn::grouped_matmul prim(layer_name, base_inputs, scale, zp, output_dt);
    if (has_scalar_zp) {
        prim.decompression_zero_point_scalar = zp_scalar_value;
    }
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v17, GroupedMatMul);
REGISTER_FACTORY_IMPL(internal, GroupedMatMulCompressed);

}  // namespace ov::intel_gpu
