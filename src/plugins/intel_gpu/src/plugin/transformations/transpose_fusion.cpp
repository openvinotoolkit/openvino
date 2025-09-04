// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose_fusion.hpp"

#include <iostream>
#include <ostream>
#include <vector>

#include "graph/include/gemm_inst.h"
#include "intel_gpu/op/gemm.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/Convolution.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/any.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/vl_sdpa.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass::pattern;
using ov::pass::pattern::op::Or;

namespace ov::intel_gpu {

namespace {

bool is_valid_order(const std::vector<size_t>& target_order, bool is_output_transpose) {
    // Check valid input/output transpose order for onednn gemm primitive
    cldnn::format fmt_dummy = cldnn::format::bfyx;
    if (is_output_transpose) {
        return cldnn::typed_primitive_inst<cldnn::gemm>::is_fusable_permute_output_order_onednn(target_order, fmt_dummy);
    } else {
        return cldnn::typed_primitive_inst<cldnn::gemm>::is_fusable_permute_input_order_onednn(target_order, fmt_dummy);
    }
}

bool has_optimized_version(const ov::Output<ov::Node>& output, bool supports_immad, bool is_output_transpose = false) {
    if (!output.get_element_type().is_real())
        return false;

    if (output.get_partial_shape().is_static() && !supports_immad)
        return false;

    auto order_node = output.get_node()->get_input_node_shared_ptr(1);
    if (!ov::is_type<ov::op::v0::Constant>(order_node))
        return false;

    auto transpose_order = ov::as_type_ptr<ov::op::v0::Constant>(order_node)->cast_vector<int64_t>();
    const auto expected_dims_num = 4;

    std::vector<size_t> order(std::begin(transpose_order), std::end(transpose_order));
    if (expected_dims_num > order.size()) {
        size_t orders_to_add = expected_dims_num - order.size();
        for (size_t i = 0; i < orders_to_add; ++i)
            order.insert(order.begin(), i);
        for (size_t i = orders_to_add; i < order.size(); ++i)
            order[i] = order[i] + orders_to_add;
    }
    
    return is_valid_order(order, is_output_transpose);
}
}  // namespace

TransposeFusion::TransposeFusion(bool supports_immad) {
    add_matcher<TransposeConv1x1TransposeMatcher>(supports_immad);
    add_matcher<TransposeMatMulTransposeMatcher>(supports_immad);
    add_matcher<TransposeMatMulMatcher>(supports_immad);
    add_matcher<TransposeSDPAMatcher>();
    add_matcher<TransposeVLSDPAMatcher>();
}

TransposeVLSDPAMatcher::TransposeVLSDPAMatcher() {
    auto is_fp_type = [](const ov::Output<ov::Node>& output) -> bool {
        switch (output.get_element_type()) {
        case ov::element::f16:
        case ov::element::f32:
            return true;
        default:
            return false;
        }
    };
    auto not_transpose = [](const ov::Output<ov::Node>& output) -> bool {
        return ov::as_type_ptr<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr;
    };

    auto input_q_m = any_input(not_transpose);
    auto input_k_m = any_input(not_transpose);
    auto input_v_m = any_input(not_transpose);
    auto input_cu_seqlens = any_input(not_transpose);

    auto transpose_q_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_k_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_v_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_q_m = wrap_type<ov::op::v1::Transpose>({input_q_m, transpose_q_order_m}, is_fp_type);
    auto transpose_k_m = wrap_type<ov::op::v1::Transpose>({input_k_m, transpose_k_order_m}, is_fp_type);
    auto transpose_v_m = wrap_type<ov::op::v1::Transpose>({input_v_m, transpose_v_order_m}, is_fp_type);

    auto sdpa_m = wrap_type<ov::op::internal::VLSDPA>({transpose_q_m, transpose_k_m, transpose_v_m, input_cu_seqlens});

    // fuse output transpose into VLSDPA too
    auto transpose_o_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_o_m = wrap_type<ov::op::v1::Transpose>({sdpa_m, transpose_o_order_m}, is_fp_type);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto sdpa = ov::as_type_ptr<ov::op::internal::VLSDPA>(pattern_map.at(sdpa_m).get_node_shared_ptr());

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
        size_t output_o_input_idx = sdpa->get_input_source_output(2).get_index();

        auto process_transpose = [](const std::shared_ptr<Node>& transpose_node,
                                    const std::shared_ptr<Node>& transpose_order_const_node,
                                    std::vector<int64_t>& order,
                                    size_t& output_idx) {
            auto transpose_order_const = ov::as_type_ptr<ov::op::v0::Constant>(transpose_order_const_node);
            std::vector<int64_t> _order = transpose_order_const->cast_vector<int64_t>();

            // Allow any transposes without head_size dim position change
            if (_order.back() != static_cast<int64_t>(_order.size() - 1))
                return false;

            auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(transpose_node);
            output_idx = transpose->get_input_source_output(0).get_index();

            order = _order;

            return true;
        };

        bool can_fuse_transposes = true;
        if (pattern_map.count(transpose_q_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_q_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_q_order_m).get_node_shared_ptr(),
                                                     order_q,
                                                     input_q_output_idx);

        if (pattern_map.count(transpose_k_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_k_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_k_order_m).get_node_shared_ptr(),
                                                     order_k,
                                                     input_k_output_idx);

        if (pattern_map.count(transpose_v_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_v_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_v_order_m).get_node_shared_ptr(),
                                                     order_v,
                                                     input_v_output_idx);

        if (pattern_map.count(transpose_o_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_o_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_o_order_m).get_node_shared_ptr(),
                                                     order_output,
                                                     output_o_input_idx);
        if (!can_fuse_transposes)
            return false;

        auto input_q = ov::Output<Node>(pattern_map.at(input_q_m).get_node_shared_ptr(), input_q_output_idx);
        auto input_k = ov::Output<Node>(pattern_map.at(input_k_m).get_node_shared_ptr(), input_k_output_idx);
        auto input_v = ov::Output<Node>(pattern_map.at(input_v_m).get_node_shared_ptr(), input_v_output_idx);

        OutputVector inputs;
        inputs.push_back(ov::Output<Node>(pattern_map.at(input_q_m).get_node_shared_ptr(), input_q_output_idx));
        inputs.push_back(ov::Output<Node>(pattern_map.at(input_k_m).get_node_shared_ptr(), input_k_output_idx));
        inputs.push_back(ov::Output<Node>(pattern_map.at(input_v_m).get_node_shared_ptr(), input_v_output_idx));
        inputs.push_back(sdpa->get_input_source_output(3));
        auto sdpa_new = std::make_shared<ov::op::internal::VLSDPA>(inputs, order_q, order_k, order_v, order_output);

        auto transpose_o = ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose_o_m).get_node_shared_ptr());
        ov::replace_node(transpose_o, sdpa_new);

        sdpa_new->set_friendly_name(transpose_o->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa_new);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_o_m, "TransposeVLSDPAMatcher");
    this->register_matcher(m, callback);
}

TransposeSDPAMatcher::TransposeSDPAMatcher() {
    auto is_fp_type = [](const ov::Output<ov::Node>& output) -> bool {
        switch (output.get_element_type()) {
        case ov::element::f16:
        case ov::element::f32:
            return true;
        default:
            return false;
        }
    };
    auto not_transpose = [is_fp_type](const ov::Output<ov::Node>& output) -> bool {
        return ov::as_type_ptr<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr && is_fp_type(output);
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

    auto sdpa_without_attn_mask_m = wrap_type<ov::op::v13::ScaledDotProductAttention>({sdpa_in_q, sdpa_in_k, sdpa_in_v});
    auto sdpa_with_attn_mask_m = wrap_type<ov::op::v13::ScaledDotProductAttention>({sdpa_in_q, sdpa_in_k, sdpa_in_v, input_attn_mask});
    auto sdpa_with_attn_mask_and_scale_m = wrap_type<ov::op::v13::ScaledDotProductAttention>({sdpa_in_q, sdpa_in_k, sdpa_in_v, input_attn_mask, input_scale});

    auto sdpa_m = std::make_shared<Or>(OutputVector{sdpa_without_attn_mask_m, sdpa_with_attn_mask_m, sdpa_with_attn_mask_and_scale_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(m.get_match_root());

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
            auto transpose_order_const = ov::as_type_ptr<ov::op::v0::Constant>(transpose_order_const_node);

            order = transpose_order_const->cast_vector<int64_t>();
            // Allow any transposes without head_size dim position change
            if (order.back() != static_cast<int64_t>(order.size() - 1))
                return false;

            auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(transpose_node);
            output_idx = transpose->get_input_source_output(0).get_index();

            return true;
        };

        bool can_fuse_transposes = true;
        if (pattern_map.count(transpose_q_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_q_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_q_order_m).get_node_shared_ptr(),
                                                     order_q,
                                                     input_q_output_idx);

        if (pattern_map.count(transpose_k_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_k_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_k_order_m).get_node_shared_ptr(),
                                                     order_k,
                                                     input_k_output_idx);

        if (pattern_map.count(transpose_v_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_v_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_v_order_m).get_node_shared_ptr(),
                                                     order_v,
                                                     input_v_output_idx);

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
        return ov::as_type_ptr<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr && output.get_element_type().is_real();
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

    auto matmul_m = wrap_type<ov::op::v0::MatMul>({matmul_in_a, matmul_in_b}, matmul_predicate);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(matmul_m).get_node_shared_ptr());
        if (!matmul || transformation_callback(matmul)) {
            return false;
        }

        auto users = matmul->get_output_target_inputs(0);
        if (users.size() == 1 && ov::as_type<ov::op::v1::Transpose>(users.begin()->get_node()) != nullptr) {
            return false;
        }

        auto order_a = op::Gemm::default_order(matmul->get_input_partial_shape(0).size());
        auto order_b = op::Gemm::default_order(matmul->get_input_partial_shape(1).size());
        auto order_c = op::Gemm::default_order(matmul->get_output_partial_shape(0).size());
        size_t input_a_output_idx = matmul->get_input_source_output(0).get_index();
        size_t input_b_output_idx = matmul->get_input_source_output(1).get_index();

        if (pattern_map.count(transpose_a_m) > 0) {
            auto tranpose_a_order = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(transpose_a_order_m).get_node_shared_ptr());
            order_a = tranpose_a_order->cast_vector<int64_t>();
            auto tranpose_a = ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose_a_m).get_node_shared_ptr());
            input_a_output_idx = tranpose_a->get_input_source_output(0).get_index();
        }
        if (matmul->get_transpose_a() && order_a.size() > 1) {
            std::swap(*(order_a.end() - 1), *(order_a.end() - 2));
        }
        if (pattern_map.count(transpose_b_m) > 0) {
            auto tranpose_b_order = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(transpose_b_order_m).get_node_shared_ptr());
            order_b = tranpose_b_order->cast_vector<int64_t>();
            auto tranpose_b = ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose_b_m).get_node_shared_ptr());
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


TransposeConv1x1TransposeMatcher::TransposeConv1x1TransposeMatcher(bool supports_immad) {
    auto static_rank_gt_1 = [](const ov::Output<ov::Node>& output) {
        const auto& r = output.get_partial_shape().rank();
        return r.is_static() && r.get_length() > 1;
    };
    auto weights_path = [&static_rank_gt_1](const ov::Output<ov::Node>& output) {
        const auto& pshape = output.get_partial_shape();
        return ov::op::util::is_on_constant_path(output) && static_rank_gt_1(output) && pshape.is_static() &&
               std::count_if(pshape.begin(), pshape.end(), [](const ov::Dimension& x) {
                   return x == 1;
               }) == 2;
    };
    auto input_transpose_predicate = [supports_immad](const ov::Output<ov::Node>& output) -> bool {
        return has_optimized_version(output, supports_immad, false);
    };
    auto output_transpose_predicate = [supports_immad](const ov::Output<ov::Node>& output) -> bool {
        return has_optimized_version(output, supports_immad, true);
    };

    
    auto first_input_m = ov::pass::pattern::any_input();
    auto a_order_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto transpose_activations_m = ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({first_input_m, a_order_m});        //, input_transpose_predicate);
    auto reshape_activations_m = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({first_input_m, a_order_m});      //, input_transpose_predicate);
    auto a_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{transpose_activations_m, reshape_activations_m});

    auto weights_m = ov::pass::pattern::any_input(weights_path);  // weights
    auto weights_scales_m = ov::pass::pattern::any_input();
    auto weights_zp_m = ov::pass::pattern::any_input();
    auto weight_convert_m = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({weights_m});
    auto weight_subtract_m = ov::pass::pattern::wrap_type<ov::op::v1::Subtract>({weight_convert_m, weights_zp_m});
    // Make zp subtraction optional to account for symmetrical quantization cases
    auto weight_dequantized_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{weight_convert_m, weight_subtract_m});
    auto weight_mult_m = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({weight_dequantized_m, weights_scales_m});
    auto conv1x1_m = ov::pass::pattern::wrap_type<ov::op::v1::Convolution>({a_m, weight_mult_m});

    auto convert_m = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({conv1x1_m});
    auto conv_out_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{conv1x1_m, convert_m});

    auto c_order_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto transpose_output_m = ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({conv_out_m, c_order_m});            //, output_transpose_predicate);
    auto reshape_output_m = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({conv_out_m, c_order_m});  //, output_transpose_predicate);
    auto output_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{transpose_output_m, reshape_output_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto conv1x1 = ov::as_type_ptr<ov::op::v1::Convolution>(pattern_map.at(conv1x1_m).get_node_shared_ptr());
        auto weight_convert = ov::as_type_ptr<ov::op::v0::Convert>(pattern_map.at(weight_convert_m).get_node_shared_ptr());
        auto weight_sub = (pattern_map.count(weight_subtract_m) > 0) ? pattern_map.at(weight_subtract_m).get_node_shared_ptr() : nullptr;
        auto weight_mult = ov::as_type_ptr<ov::op::v1::Multiply>(pattern_map.at(weight_mult_m).get_node_shared_ptr());
        auto convert_out = (pattern_map.count(convert_m) > 0) ? pattern_map.at(convert_m).get_node_shared_ptr() : nullptr;
        if (!conv1x1 || transformation_callback(conv1x1)) {
            return false;
        }

        auto weight = pattern_map.at(weights_m).get_node_shared_ptr();
        auto scale = pattern_map.at(weights_scales_m).get_node_shared_ptr();
        auto zp = (pattern_map.count(weights_zp_m) > 0) ? pattern_map.at(weights_zp_m).get_node_shared_ptr() : nullptr;
        auto activation = pattern_map.at(first_input_m).get_node_shared_ptr();        

        // add reshape after weight 9216 x 3072 x 1 x --> 9216 x 3072
        std::vector<int> values_reshape_b;
        auto shape_b = weight->get_output_partial_shape(0);
        for (auto i = 0; i < shape_b.size(); i++)
            if (shape_b.to_shape()[i] != 1) {
                values_reshape_b.push_back(shape_b.to_shape()[i]);
            }
        auto reshape_weight_const = ov::op::v0::Constant::create(element::i32, Shape{2}, values_reshape_b);  //{9216, 3072});
        auto Reshape_weight = std::make_shared<ov::op::v1::Reshape>(weight, reshape_weight_const, false);
        MatcherPass::register_new_node(Reshape_weight);
        Reshape_weight->set_friendly_name(weight->get_friendly_name() + "_Reshape_weight");
        // FixMe: this is a point of interest - it protects quantized weights from being inflated by constant-folded conversion
        // ov::disable_constant_folding(Reshape_weight);
        auto weight_squeezed_convert = weight_convert->clone_with_new_inputs({Reshape_weight});
        ov::disable_constant_folding(weight_squeezed_convert);

        // add reshape after scales
        std::vector<int> values_reshape_bs;
        auto shape_bs = scale->get_output_partial_shape(0);
        for (auto i = 0; i < shape_bs.size(); i++)
            if (shape_bs.to_shape()[i] != 1) {
                values_reshape_bs.push_back(shape_bs.to_shape()[i]);
            }
        // Add broadcast dimensions
        while (values_reshape_bs.size() < 2) {
            values_reshape_bs.push_back(1);
        }
        auto reshape_scale_const = ov::op::v0::Constant::create(element::i32, Shape{2}, values_reshape_bs);  //{9216, 3072});
        auto Reshape_scale = std::make_shared<ov::op::v1::Reshape>(scale, reshape_scale_const, false);
        MatcherPass::register_new_node(Reshape_scale);
        Reshape_scale->set_friendly_name(scale->get_friendly_name() + "_Reshape_scale");

        auto scaled_weight = weight_mult->clone_with_new_inputs({weight_squeezed_convert, Reshape_scale});
        if (zp) {
            // add reshape after zero points
            std::vector<int> values_reshape_zp;
            auto shape_zp = zp->get_output_partial_shape(0);
            for (auto i = 0; i < shape_zp.size(); i++)
                if (shape_zp.to_shape()[i] != 1) {
                    values_reshape_zp.push_back(shape_zp.to_shape()[i]);
                }
            // Add broadcast dimensions
            while (values_reshape_zp.size() < 2) {
                values_reshape_zp.push_back(1);
            }
            auto reshape_zp_const = ov::op::v0::Constant::create(element::i32, Shape{2}, values_reshape_zp);  //{9216, 3072});
            auto Reshape_zp = std::make_shared<ov::op::v1::Reshape>(zp, reshape_zp_const, false);
            MatcherPass::register_new_node(Reshape_zp);
            Reshape_zp->set_friendly_name(zp->get_friendly_name() + "_Reshape_zp");
            auto zero_adjusted_weight = weight_sub->clone_with_new_inputs({weight_squeezed_convert, Reshape_zp});
            scaled_weight = weight_mult->clone_with_new_inputs({zero_adjusted_weight, Reshape_scale});
        }
        ov::disable_constant_folding(scaled_weight);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(activation, scaled_weight, false, true);
        // ToDo: handle out conversion if it exists
        if (convert_out) {
            auto convert_final = convert_out->clone_with_new_inputs({matmul});
            convert_final->set_friendly_name(m.get_match_root()->get_friendly_name());
            ov::copy_runtime_info(m.get_matched_nodes(), convert_final);
            ov::replace_node(m.get_match_root(), convert_final);
        } else {
            matmul->set_friendly_name(m.get_match_root()->get_friendly_name());
            ov::copy_runtime_info(m.get_matched_nodes(), matmul);
            ov::replace_node(m.get_match_root(), matmul);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(output_m, "TransposeConv1x1TransposeMatcher");
    this->register_matcher(m, callback);
}

TransposeMatMulTransposeMatcher::TransposeMatMulTransposeMatcher(bool supports_immad) {
    auto not_transpose = [](const ov::Output<ov::Node>& output) -> bool {
        return ov::as_type_ptr<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr && output.get_element_type().is_real();
    };
    auto input_transpose_predicate = [supports_immad](const ov::Output<ov::Node>& output) -> bool {
        return has_optimized_version(output, supports_immad, false);
    };
    auto output_transpose_predicate = [supports_immad](const ov::Output<ov::Node>& output) -> bool {
        return has_optimized_version(output, supports_immad, true);
    };
    auto input_a_m = any_input(not_transpose);
    auto input_b_m = any_input(not_transpose);
    auto transpose_a_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_b_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_a_m = wrap_type<ov::op::v1::Transpose>({input_a_m, transpose_a_order_m}, input_transpose_predicate);
    auto transpose_b_m = wrap_type<ov::op::v1::Transpose>({input_b_m, transpose_b_order_m}, input_transpose_predicate);

    auto matmul_in_a = std::make_shared<Or>(OutputVector{input_a_m, transpose_a_m});
    auto matmul_in_b = std::make_shared<Or>(OutputVector{input_b_m, transpose_b_m});

    auto matmul_m = wrap_type<ov::op::v0::MatMul>({matmul_in_a, matmul_in_b}, consumers_count(1));
    auto transpose_c_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_c_m = wrap_type<ov::op::v1::Transpose>({matmul_m, transpose_c_order_m}, output_transpose_predicate);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(matmul_m).get_node_shared_ptr());
        if (!matmul || transformation_callback(matmul)) {
            return false;
        }

        auto tranpose_c_order = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(transpose_c_order_m).get_node_shared_ptr());
        auto order_a = op::Gemm::default_order(matmul->get_input_partial_shape(0).size());
        auto order_b = op::Gemm::default_order(matmul->get_input_partial_shape(1).size());
        auto order_c = tranpose_c_order->cast_vector<int64_t>();
        size_t input_a_output_idx = matmul->get_input_source_output(0).get_index();
        size_t input_b_output_idx = matmul->get_input_source_output(1).get_index();

        if (pattern_map.count(transpose_a_m) > 0) {
            auto tranpose_a_order = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(transpose_a_order_m).get_node_shared_ptr());
            order_a = tranpose_a_order->cast_vector<int64_t>();
            auto tranpose_a = ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose_a_m).get_node_shared_ptr());
            input_a_output_idx = tranpose_a->get_input_source_output(0).get_index();
        }
        if (matmul->get_transpose_a() && order_a.size() > 1) {
            std::swap(*(order_a.end() - 1), *(order_a.end() - 2));
        }
        if (pattern_map.count(transpose_b_m) > 0) {
            auto tranpose_b_order = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(transpose_b_order_m).get_node_shared_ptr());
            order_b = tranpose_b_order->cast_vector<int64_t>();
            auto tranpose_b = ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose_b_m).get_node_shared_ptr());
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

}  // namespace ov::intel_gpu