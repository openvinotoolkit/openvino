// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast_mul_reduce_to_matmul.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

BroadcastMulReduceToMatMul::BroadcastMulReduceToMatMul() {
    using namespace ov::pass::pattern;

    // Match: any_input -> Multiply -> ReduceSum
    // Validate that Multiply inputs are Unsqueeze in the callback.
    auto multiply_in_a = any_input();
    auto multiply_in_b = any_input();
    auto multiply_m = wrap_type<ov::op::v1::Multiply>({multiply_in_a, multiply_in_b});
    auto reduce_axis_m = wrap_type<ov::op::v0::Constant>();
    auto reduce_sum_m = wrap_type<ov::op::v1::ReduceSum>({multiply_m, reduce_axis_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto reduce_sum = ov::as_type_ptr<ov::op::v1::ReduceSum>(m.get_match_root());
        auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(
            pattern_map.at(multiply_m).get_node_shared_ptr());

        if (!reduce_sum || !multiply)
            return false;

        // ReduceSum must have keep_dims=false
        if (reduce_sum->get_keep_dims())
            return false;

        // Multiply should have exactly one consumer (the ReduceSum)
        if (multiply->get_output_target_inputs(0).size() != 1)
            return false;

        // Both Multiply inputs must be Unsqueeze
        auto unsqueeze_a = ov::as_type_ptr<ov::op::v0::Unsqueeze>(multiply->get_input_node_shared_ptr(0));
        auto unsqueeze_b = ov::as_type_ptr<ov::op::v0::Unsqueeze>(multiply->get_input_node_shared_ptr(1));

        if (!unsqueeze_a || !unsqueeze_b)
            return false;

        // Get unsqueeze axes (must be constant single values)
        auto axis_a_const = ov::as_type_ptr<ov::op::v0::Constant>(unsqueeze_a->get_input_node_shared_ptr(1));
        auto axis_b_const = ov::as_type_ptr<ov::op::v0::Constant>(unsqueeze_b->get_input_node_shared_ptr(1));
        auto axis_r_const = ov::as_type_ptr<ov::op::v0::Constant>(reduce_sum->get_input_node_shared_ptr(1));

        if (!axis_a_const || !axis_b_const || !axis_r_const)
            return false;

        auto axis_a_vals = axis_a_const->cast_vector<int64_t>();
        auto axis_b_vals = axis_b_const->cast_vector<int64_t>();
        auto axis_r_vals = axis_r_const->cast_vector<int64_t>();

        // Each must be a single axis
        if (axis_a_vals.size() != 1 || axis_b_vals.size() != 1 || axis_r_vals.size() != 1)
            return false;

        // Get the rank of the expanded intermediate from the Multiply output.
        // Each input is R dimensional; after two Unsqueezes, the Multiply output is (R+1) dimensional.
        auto mul_out_pshape = multiply->get_output_partial_shape(0);
        if (mul_out_pshape.rank().is_dynamic())
            return false;

        const int64_t rank_expanded = mul_out_pshape.rank().get_length();

        // Normalize negative axes
        int64_t axis_a = axis_a_vals[0];
        int64_t axis_b = axis_b_vals[0];
        int64_t axis_r = axis_r_vals[0];
        if (axis_a < 0) axis_a += rank_expanded;
        if (axis_b < 0) axis_b += rank_expanded;
        if (axis_r < 0) axis_r += rank_expanded;

        // Validate axes are in range and all different
        if (axis_a < 0 || axis_a >= rank_expanded ||
            axis_b < 0 || axis_b >= rank_expanded ||
            axis_r < 0 || axis_r >= rank_expanded)
            return false;

        if (axis_a == axis_b || axis_a == axis_r || axis_b == axis_r)
            return false;

        // Need at least 3D intermediate
        if (rank_expanded < 3)
            return false;

        // Compute batch axes in the expanded (R+1) dim space.
        // In MatMul terms: axis_b maps to M, axis_a maps to N, axis_r maps to K.
        // Everything else is a batch dimension.
        std::vector<int64_t> batch_axes;
        for (int64_t d = 0; d < rank_expanded; d++) {
            if (d != axis_a && d != axis_b && d != axis_r)
                batch_axes.push_back(d);
        }

        // Map expanded (R+1) dim axis to A's original R dim axis (A is missing axis_a)
        auto map_to_a = [axis_a](int64_t d_exp) -> int64_t {
            return d_exp < axis_a ? d_exp : d_exp - 1;
        };
        // Map expanded (R+1) dim axis to B's original R dim axis (B is missing axis_b)
        auto map_to_b = [axis_b](int64_t d_exp) -> int64_t {
            return d_exp < axis_b ? d_exp : d_exp - 1;
        };

        // Transpose order for A: [...batch, M, K]
        // M = axis unique to A (axis_b position in expanded space)
        // K = reduced axis (axis_r, summed out by ReduceSum)
        std::vector<int64_t> transpose_a;
        for (auto bd : batch_axes)
            transpose_a.push_back(map_to_a(bd));
        transpose_a.push_back(map_to_a(axis_b));   // M (axis unique to A)
        transpose_a.push_back(map_to_a(axis_r));   // K (reduced axis)

        // Transpose order for B: [...batch, N, K]
        // N = axis unique to B (axis_a position in expanded space)
        // K = reduced axis (axis_r, summed out by ReduceSum)
        std::vector<int64_t> transpose_b;
        for (auto bd : batch_axes)
            transpose_b.push_back(map_to_b(bd));
        transpose_b.push_back(map_to_b(axis_a));   // N (axis unique to B)
        transpose_b.push_back(map_to_b(axis_r));   // K (reduced axis)

        // Output transpose: MatMul output [...batch, M, N] expected ReduceSum output order.
        // Axis labels below use expanded (R+1) dim numbering for bookkeeping.
        std::vector<int64_t> matmul_output_order;
        for (auto bd : batch_axes)
            matmul_output_order.push_back(bd);
        matmul_output_order.push_back(axis_b);
        matmul_output_order.push_back(axis_a);

        std::vector<int64_t> expected_output_order;
        for (int64_t d = 0; d < rank_expanded; d++) {
            if (d != axis_r)
                expected_output_order.push_back(d);
        }

        const int64_t rank_input = rank_expanded - 1;

        std::vector<int64_t> transpose_out(rank_input);
        for (int64_t i = 0; i < rank_input; i++) {
            for (int64_t j = 0; j < rank_input; j++) {
                if (matmul_output_order[j] == expected_output_order[i]) {
                    transpose_out[i] = j;
                    break;
                }
            }
        }

        auto is_identity = [](const std::vector<int64_t>& perm) {
            for (size_t i = 0; i < perm.size(); i++) {
                if (perm[i] != static_cast<int64_t>(i))
                    return false;
            }
            return true;
        };

        auto input_a = unsqueeze_a->input_value(0);
        auto input_b = unsqueeze_b->input_value(0);

        ov::NodeVector new_nodes;

        // Transpose A if needed: [...batch, M, K]
        ov::Output<ov::Node> a_prepared = input_a;
        if (!is_identity(transpose_a)) {
            auto perm_a = ov::op::v0::Constant::create(ov::element::i64, {transpose_a.size()}, transpose_a);
            auto tr_a = std::make_shared<ov::op::v1::Transpose>(input_a, perm_a);
            new_nodes.push_back(tr_a);
            a_prepared = tr_a->output(0);
        }

        // Transpose B if needed: [...batch, N, K]
        ov::Output<ov::Node> b_prepared = input_b;
        if (!is_identity(transpose_b)) {
            auto perm_b = ov::op::v0::Constant::create(ov::element::i64, {transpose_b.size()}, transpose_b);
            auto tr_b = std::make_shared<ov::op::v1::Transpose>(input_b, perm_b);
            new_nodes.push_back(tr_b);
            b_prepared = tr_b->output(0);
        }

        // MatMul with transpose_b=true (batched):
        // A: [...batch, M, K]
        // B: [...batch, N, K]  →  transpose_b swaps last 2: [...batch, K, N]
        // Result: [...batch, M, N]
        auto matmul = std::make_shared<ov::op::v0::MatMul>(a_prepared, b_prepared, false, true);
        new_nodes.push_back(matmul);

        ov::Output<ov::Node> result = matmul->output(0);

        // Transpose output to expected order if needed
        if (!is_identity(transpose_out)) {
            auto perm_out = ov::op::v0::Constant::create(ov::element::i64, {transpose_out.size()}, transpose_out);
            auto tr_out = std::make_shared<ov::op::v1::Transpose>(result, perm_out);
            new_nodes.push_back(tr_out);
            result = tr_out->output(0);
        }

        // Ensure output data type matches the original ReduceSum output.
        // GPU Gemm may compute in f32 while the graph expects f16.
        auto expected_et = reduce_sum->get_output_element_type(0);
        if (result.get_element_type() != expected_et) {
            auto cvt = std::make_shared<ov::op::v0::Convert>(result, expected_et);
            new_nodes.push_back(cvt);
            result = cvt->output(0);
        }

        result.get_node_shared_ptr()->set_friendly_name(reduce_sum->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), new_nodes);
        ov::replace_node(reduce_sum, result.get_node_shared_ptr());

        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(reduce_sum_m, "BroadcastMulReduceToMatMul");
    this->register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
