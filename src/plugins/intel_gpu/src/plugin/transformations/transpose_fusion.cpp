// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/gemm.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "transpose_fusion.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/any.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass::pattern;
using ov::pass::pattern::op::Or;

namespace ov {
namespace intel_gpu {

namespace {

bool has_optimized_version(const ov::Output<ov::Node>& output, bool supports_immad) {
    if (!output.get_element_type().is_real())
        return false;

    if (output.get_partial_shape().is_static() && !supports_immad)
        return false;

    auto order_node = output.get_node()->get_input_node_shared_ptr(1);
    if (!ov::is_type<ov::op::v0::Constant>(order_node))
        return false;

    auto transpose_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(order_node)->cast_vector<int64_t>();
    static const std::vector<std::vector<int64_t>> allowed_orders = {
        {0, 1, 2, 3},
        {0, 1, 3, 2},
        {0, 2, 1, 3},
        {0, 3, 1, 2},
        {1, 2, 0, 3},
        {2, 0, 1, 3},
        {3, 0, 1, 2},
    };

    const auto expected_dims_num = 4;
    const auto original_dims_num = transpose_order.size();
    if (original_dims_num < expected_dims_num) {
        transpose_order.resize(expected_dims_num);
        std::iota(transpose_order.begin() + original_dims_num, transpose_order.end(), original_dims_num);
    }

    if (!cldnn::one_of(transpose_order, allowed_orders))
        return false;

    return true;
}
}  // namespace

TransposeFusion::TransposeFusion(bool supports_immad) {
    add_matcher<TransposeMatMulTransposeMatcher>(supports_immad);
    add_matcher<TransposeMatMulMatcher>(supports_immad);
    add_matcher<TransposeSDPAMatcher>();
}

TransposeSDPAMatcher::TransposeSDPAMatcher() {
    auto is_fp_type = [](const ov::Output<ov::Node>& output) -> bool {
        switch (output.get_element_type()) {
            case ov::element::f16:
            case ov::element::f32: return true;
            default: return false;
        }
    };
    auto not_transpose = [is_fp_type](const ov::Output<ov::Node>& output) -> bool {
        return std::dynamic_pointer_cast<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr
               && is_fp_type(output);
    };

    auto input_q_m = any_input(not_transpose);
    auto input_k_m = any_input(not_transpose);
    auto input_v_m = any_input(not_transpose);
    auto input_attn_mask = any_input(not_transpose);
    auto input_scale = any_input(not_transpose);
    auto transpose_q_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_k_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_v_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_q_m = wrap_type<ov::op::v1::Transpose>({input_q_m, transpose_q_order_m}, is_fp_type);
    auto transpose_k_m = wrap_type<ov::op::v1::Transpose>({input_k_m, transpose_k_order_m}, is_fp_type);
    auto transpose_v_m = wrap_type<ov::op::v1::Transpose>({input_v_m, transpose_v_order_m}, is_fp_type);

    auto sdpa_in_q = std::make_shared<Or>(OutputVector{input_q_m, transpose_q_m});
    auto sdpa_in_k = std::make_shared<Or>(OutputVector{input_k_m, transpose_k_m});
    auto sdpa_in_v = std::make_shared<Or>(OutputVector{input_v_m, transpose_v_m});

    auto sdpa_without_attn_mask_m = wrap_type<ov::op::v13::ScaledDotProductAttention>({ sdpa_in_q, sdpa_in_k, sdpa_in_v });
    auto sdpa_with_attn_mask_m = wrap_type<ov::op::v13::ScaledDotProductAttention>({ sdpa_in_q, sdpa_in_k, sdpa_in_v, input_attn_mask });
    auto sdpa_with_attn_mask_and_scale_m =
        wrap_type<ov::op::v13::ScaledDotProductAttention>({ sdpa_in_q, sdpa_in_k, sdpa_in_v, input_attn_mask, input_scale });

    auto sdpa_m = std::make_shared<Or>(OutputVector{sdpa_without_attn_mask_m, sdpa_with_attn_mask_m, sdpa_with_attn_mask_and_scale_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto sdpa = std::dynamic_pointer_cast<ov::op::v13::ScaledDotProductAttention>(m.get_match_root());

        if (!sdpa || transformation_callback(sdpa)) {
            return false;
        }

        auto order_q = op::SDPA::default_order(sdpa->get_input_partial_shape(0).size());
        auto order_k = op::SDPA::default_order(sdpa->get_input_partial_shape(1).size());
        auto order_v = op::SDPA::default_order(sdpa->get_input_partial_shape(2).size());
        auto order_output = op::SDPA::default_order(sdpa->get_output_partial_shape(0).size());
        size_t input_q_output_idx = sdpa->get_input_source_output(0).get_index();
        size_t input_k_output_idx = sdpa->get_input_source_output(1).get_index();
        size_t input_v_output_idx = sdpa->get_input_source_output(2).get_index();

        auto process_transpose = [](const std::shared_ptr<Node>& transpose_node,
                                    const std::shared_ptr<Node>& transpose_order_const_node,
                                    std::vector<int64_t>& order,
                                    size_t& output_idx) {
            auto transpose_order_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(transpose_order_const_node);

            order = transpose_order_const->cast_vector<int64_t>();
            // Allow any transposes without head_size dim position change
            if (order.back() != static_cast<int64_t>(order.size() - 1))
                return false;

            auto transpose = std::dynamic_pointer_cast<ov::op::v1::Transpose>(transpose_node);
            output_idx = transpose->get_input_source_output(0).get_index();

            return true;
        };

        bool can_fuse_transposes = true;
        if (pattern_map.count(transpose_q_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_q_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_q_order_m).get_node_shared_ptr(),
                                                     order_q, input_q_output_idx);

        if (pattern_map.count(transpose_k_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_k_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_k_order_m).get_node_shared_ptr(),
                                                     order_k, input_k_output_idx);

        if (pattern_map.count(transpose_v_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_v_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_v_order_m).get_node_shared_ptr(),
                                                     order_v, input_v_output_idx);

        if (!can_fuse_transposes)
            return false;

        auto input_q = ov::Output<Node>(pattern_map.at(input_q_m).get_node_shared_ptr(), input_q_output_idx);
        auto input_k = ov::Output<Node>(pattern_map.at(input_k_m).get_node_shared_ptr(), input_k_output_idx);
        auto input_v = ov::Output<Node>(pattern_map.at(input_v_m).get_node_shared_ptr(), input_v_output_idx);

        OutputVector inputs;
        inputs.push_back(ov::Output<Node>(pattern_map.at(input_q_m).get_node_shared_ptr(), input_q_output_idx));
        inputs.push_back(ov::Output<Node>(pattern_map.at(input_k_m).get_node_shared_ptr(), input_k_output_idx));
        inputs.push_back(ov::Output<Node>(pattern_map.at(input_v_m).get_node_shared_ptr(), input_v_output_idx));

        if (pattern_map.find(sdpa_with_attn_mask_m) != pattern_map.end()) {
            inputs.push_back(sdpa->get_input_source_output(3));
        } else if (pattern_map.find(sdpa_with_attn_mask_and_scale_m) != pattern_map.end()) {
            inputs.push_back(sdpa->get_input_source_output(3));
            inputs.push_back(sdpa->get_input_source_output(4));
        }

        auto sdpa_new = std::make_shared<op::SDPA>(inputs, sdpa->get_causal(), order_q, order_k, order_v, order_output);

        sdpa_new->set_friendly_name(sdpa->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa_new);
        ov::replace_node(sdpa, sdpa_new);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdpa_m, "TransposeSDPAMatcher");
    this->register_matcher(m, callback);
}

TransposeMatMulMatcher::TransposeMatMulMatcher(bool supports_immad) {
    auto not_transpose = [](const ov::Output<ov::Node>& output) -> bool {
        return std::dynamic_pointer_cast<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr
               && output.get_element_type().is_real();
    };

    auto transpose_predicate = [supports_immad](const ov::Output<ov::Node>& output) -> bool {
        return has_optimized_version(output, supports_immad);
    };

    // Don't convert MatMul -> Gemm if no transpose input found as
    // CreateMatMulOp factory can now insert extra transpose which improves the performance
    auto matmul_predicate = [](const ov::Output<ov::Node>& output) -> bool {
        auto node = output.get_node();
        if (node->is_dynamic())
            return true;

        for (size_t i = 0; i < node->get_input_size(); i++) {
            if (ov::is_type<ov::op::v1::Transpose>(node->get_input_node_ptr(i)))
                return true;
        }

        return false;
    };

    auto input_a_m = any_input(not_transpose);
    auto input_b_m = any_input(not_transpose);
    auto transpose_a_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_b_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_a_m = wrap_type<ov::op::v1::Transpose>({input_a_m, transpose_a_order_m}, transpose_predicate);
    auto transpose_b_m = wrap_type<ov::op::v1::Transpose>({input_b_m, transpose_b_order_m}, transpose_predicate);

    auto matmul_in_a = std::make_shared<Or>(OutputVector{input_a_m, transpose_a_m});
    auto matmul_in_b = std::make_shared<Or>(OutputVector{input_b_m, transpose_b_m});

    auto matmul_m = wrap_type<ov::op::v0::MatMul>({ matmul_in_a, matmul_in_b }, matmul_predicate);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(pattern_map.at(matmul_m).get_node_shared_ptr());
        if (!matmul || transformation_callback(matmul)) {
            return false;
        }

        auto users = matmul->get_output_target_inputs(0);
        if (users.size() == 1 && dynamic_cast<ov::op::v1::Transpose*>(users.begin()->get_node()) != nullptr) {
            return false;
        }

        auto order_a = op::Gemm::default_order(matmul->get_input_partial_shape(0).size());
        auto order_b = op::Gemm::default_order(matmul->get_input_partial_shape(1).size());
        auto order_c = op::Gemm::default_order(matmul->get_output_partial_shape(0).size());
        size_t input_a_output_idx = matmul->get_input_source_output(0).get_index();
        size_t input_b_output_idx = matmul->get_input_source_output(1).get_index();

        if (pattern_map.count(transpose_a_m) > 0) {
            auto tranpose_a_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_a_order_m).get_node_shared_ptr());
            order_a = tranpose_a_order->cast_vector<int64_t>();
            auto tranpose_a = std::dynamic_pointer_cast<ov::op::v1::Transpose>(pattern_map.at(transpose_a_m).get_node_shared_ptr());
            input_a_output_idx = tranpose_a->get_input_source_output(0).get_index();
        }
        if (matmul->get_transpose_a() && order_a.size() > 1) {
            std::swap(*(order_a.end() - 1), *(order_a.end() - 2));
        }
        if (pattern_map.count(transpose_b_m) > 0) {
            auto tranpose_b_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_b_order_m).get_node_shared_ptr());
            order_b = tranpose_b_order->cast_vector<int64_t>();
            auto tranpose_b = std::dynamic_pointer_cast<ov::op::v1::Transpose>(pattern_map.at(transpose_b_m).get_node_shared_ptr());
            input_b_output_idx = tranpose_b->get_input_source_output(0).get_index();
        }
        if (matmul->get_transpose_b() && order_b.size() > 1) {
            std::swap(*(order_b.end() - 1), *(order_b.end() - 2));
        }

        auto input_a = ov::Output<Node>(pattern_map.at(input_a_m).get_node_shared_ptr(), input_a_output_idx);
        auto input_b = ov::Output<Node>(pattern_map.at(input_b_m).get_node_shared_ptr(), input_b_output_idx);

        auto gemm = std::make_shared<op::Gemm>(input_a, input_b, order_a, order_b, order_c);
        gemm->set_friendly_name(matmul->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), gemm);
        ov::replace_node(matmul, gemm);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul_m, "TransposeMatMulMatcher");
    this->register_matcher(m, callback);
}

TransposeMatMulTransposeMatcher::TransposeMatMulTransposeMatcher(bool supports_immad) {
    auto not_transpose = [](const ov::Output<ov::Node>& output) -> bool {
        return std::dynamic_pointer_cast<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr
               && output.get_element_type().is_real();
    };
    auto transpose_predicate = [supports_immad](const ov::Output<ov::Node>& output) -> bool {
        return has_optimized_version(output, supports_immad);
    };
    auto input_a_m = any_input(not_transpose);
    auto input_b_m = any_input(not_transpose);
    auto transpose_a_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_b_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_a_m = wrap_type<ov::op::v1::Transpose>({input_a_m, transpose_a_order_m}, transpose_predicate);
    auto transpose_b_m = wrap_type<ov::op::v1::Transpose>({input_b_m, transpose_b_order_m}, transpose_predicate);

    auto matmul_in_a = std::make_shared<Or>(OutputVector{input_a_m, transpose_a_m});
    auto matmul_in_b = std::make_shared<Or>(OutputVector{input_b_m, transpose_b_m});

    auto matmul_m = wrap_type<ov::op::v0::MatMul>({ matmul_in_a, matmul_in_b });
    auto transpose_c_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_c_m = wrap_type<ov::op::v1::Transpose>({matmul_m, transpose_c_order_m}, transpose_predicate);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(pattern_map.at(matmul_m).get_node_shared_ptr());
        if (!matmul || transformation_callback(matmul)) {
            return false;
        }

        auto tranpose_c_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_c_order_m).get_node_shared_ptr());
        auto order_a = op::Gemm::default_order(matmul->get_input_partial_shape(0).size());
        auto order_b = op::Gemm::default_order(matmul->get_input_partial_shape(1).size());
        auto order_c = tranpose_c_order->cast_vector<int64_t>();
        size_t input_a_output_idx = matmul->get_input_source_output(0).get_index();
        size_t input_b_output_idx = matmul->get_input_source_output(1).get_index();

        if (pattern_map.count(transpose_a_m) > 0) {
            auto tranpose_a_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_a_order_m).get_node_shared_ptr());
            order_a = tranpose_a_order->cast_vector<int64_t>();
            auto tranpose_a = std::dynamic_pointer_cast<ov::op::v1::Transpose>(pattern_map.at(transpose_a_m).get_node_shared_ptr());
            input_a_output_idx = tranpose_a->get_input_source_output(0).get_index();
        }
        if (matmul->get_transpose_a() && order_a.size() > 1) {
            std::swap(*(order_a.end() - 1), *(order_a.end() - 2));
        }
        if (pattern_map.count(transpose_b_m) > 0) {
            auto tranpose_b_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_b_order_m).get_node_shared_ptr());
            order_b = tranpose_b_order->cast_vector<int64_t>();
            auto tranpose_b = std::dynamic_pointer_cast<ov::op::v1::Transpose>(pattern_map.at(transpose_b_m).get_node_shared_ptr());
            input_b_output_idx = tranpose_b->get_input_source_output(0).get_index();
        }
        if (matmul->get_transpose_b() && order_b.size() > 1) {
            std::swap(*(order_b.end() - 1), *(order_b.end() - 2));
        }

        auto input_a = ov::Output<Node>(pattern_map.at(input_a_m).get_node_shared_ptr(), input_a_output_idx);
        auto input_b = ov::Output<Node>(pattern_map.at(input_b_m).get_node_shared_ptr(), input_b_output_idx);

        auto gemm = std::make_shared<op::Gemm>(input_a, input_b, order_a, order_b, order_c);
        gemm->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), gemm);
        ov::replace_node(m.get_match_root(), gemm);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_c_m, "TransposeMatMulTransposeMatcher");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
