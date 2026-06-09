// Copyright (C) 2018-2026 Intel Corporation
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
#include "openvino/core/graph_util.hpp"
#include "graph/include/gemm_inst.h"

#include "ov_ops/vl_sdpa.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/split.hpp"

#include <iostream>
#include <vector>
#include <ostream>

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
    add_matcher<TransposeMatMulTransposeMatcher>(supports_immad);
    add_matcher<TransposeMatMulMatcher>(supports_immad);
    add_matcher<TransposeSDPAMatcher>();
    add_matcher<TransposeVLSDPAMatcher>();
    add_matcher<QKVSplitReshapeMatcher>();
}

TransposeVLSDPAMatcher::TransposeVLSDPAMatcher() {
    auto is_fp_type = [](const ov::Output<ov::Node>& output) -> bool {
        switch (output.get_element_type()) {
            case ov::element::f16:
            case ov::element::f32: return true;
            default: return false;
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

    auto sdpa_m = wrap_type<ov::op::internal::VLSDPA>({ transpose_q_m, transpose_k_m, transpose_v_m, input_cu_seqlens });

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
            std::vector<int64_t>_order = transpose_order_const->cast_vector<int64_t>();

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
                                                     order_q, input_q_output_idx);

        if (pattern_map.count(transpose_k_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_k_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_k_order_m).get_node_shared_ptr(),
                                                     order_k, input_k_output_idx);

        if (pattern_map.count(transpose_v_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_v_m).get_node_shared_ptr(),
                                                     pattern_map.at(transpose_v_order_m).get_node_shared_ptr(),
                                                     order_v, input_v_output_idx);

        if (pattern_map.count(transpose_o_m) > 0)
            can_fuse_transposes &= process_transpose(pattern_map.at(transpose_o_m).get_node_shared_ptr(),
                                                    pattern_map.at(transpose_o_order_m).get_node_shared_ptr(),
                                                    order_output, output_o_input_idx);
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
            case ov::element::f32: return true;
            default: return false;
        }
    };
    auto not_transpose = [is_fp_type](const ov::Output<ov::Node>& output) -> bool {
        return ov::as_type_ptr<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr
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
        return ov::as_type_ptr<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr
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

TransposeMatMulTransposeMatcher::TransposeMatMulTransposeMatcher(bool supports_immad) {
    auto not_transpose = [](const ov::Output<ov::Node>& output) -> bool {
        return ov::as_type_ptr<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr
               && output.get_element_type().is_real();
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

    auto matmul_m = wrap_type<ov::op::v0::MatMul>({ matmul_in_a, matmul_in_b }, consumers_count(1));
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


// ===========================================================================
// QKVSplitReshapeMatcher
//
// Matches the following sub-graph produced by VIT-style attention blocks:
//
//   FC:  [?, ?, 4224]
//     → Reshape([0,0,3,H,S]) → [?,?,3,H,S]
//     → Transpose([2,0,3,1,4])  → [3,?,H,?,S]
//     → Split(axis=0, num=3)    → [1,?,H,?,S] × 3
//     → Squeeze(axis=0) × 3    → [?,H,?,S]   × 3
//     → Transpose([0,2,1,3])   → [?,?,H,S]   × 3  (Q, K, V paths)
//     → flatten_Reshape([0,0,H*S])            × 3
//     → downstream (RMSNorm / SDPA)
//
// Replaces with:
//   FC:  [?, ?, 4224]
//     → Reshape([0,0,3,H,S]) → [?,?,3,H,S]    (kept as-is)
//     → Split(axis=2, num=3)   → [?,?,1,H,S] × 3
//     → Squeeze(axis=2) × 3   → [?,?,H,S]   × 3
//     → flatten_Reshape([0,0,H*S])            × 3  (Transpose([0,2,1,3]) removed)
//     → downstream (unchanged)
//
// Effect:  crop_axis = 2,  reshape_axis = 2 - 1 = 1 >= 0
//   → prepare_buffer_fusing.cpp existing axis==1 path handles the in-place crop offset
//   → crop in-place optimization enabled even with dynamic sequence dimension
// ===========================================================================
QKVSplitReshapeMatcher::QKVSplitReshapeMatcher() {
    // NOTE: ov::op::v0::Squeeze is lowered to ov::op::v1::Reshape before TransposeFusion runs.
    auto transpose_m = wrap_type<ov::op::v1::Transpose>({any_input(), wrap_type<ov::op::v0::Constant>()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(
            pattern_map.at(transpose_m).get_node_shared_ptr());
        if (!transpose) {
            std::cout << "[QKVSplitReshapeMatcher] PRE: null transpose\n" << std::flush;
            return false;
        }
        const bool tc = transformation_callback(transpose);
        std::cout << "[QKVSplitReshapeMatcher] PRE: " << transpose->get_friendly_name()
                  << " tc=" << tc << "\n" << std::flush;
        if (tc)
            return false;

        std::cout << "[QKVSplitReshapeMatcher] ENTRY: " << transpose->get_friendly_name()
                  << " in=" << transpose->get_input_partial_shape(0)
                  << " out=" << transpose->get_output_partial_shape(0) << "\n" << std::flush;

        // [Check 1] Permute order must be [2,0,3,1,4]
        auto order_const = ov::as_type_ptr<ov::op::v0::Constant>(
            transpose->get_input_node_shared_ptr(1));
        if (!order_const) {
            std::cout << "[QKVSplitReshapeMatcher] FAIL Check1a: no order_const\n" << std::flush;
            return false;
        }
        const auto order = order_const->cast_vector<int64_t>();
        if (order != std::vector<int64_t>{2, 0, 3, 1, 4}) {
            std::cout << "[QKVSplitReshapeMatcher] FAIL Check1b: wrong order\n" << std::flush;
            return false;
        }

        // [Check 2] Input must be Reshape: rank-5, dim[2]==3, dim[3/4] static
        auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(
            transpose->get_input_node_shared_ptr(0));
        if (!reshape) {
            std::cout << "[QKVSplitReshapeMatcher] FAIL Check2a: input not Reshape\n" << std::flush;
            return false;
        }
        const auto& reshape_shape = reshape->get_output_partial_shape(0);
        std::cout << "[QKVSplitReshapeMatcher] reshape_shape=" << reshape_shape << "\n" << std::flush;
        if (reshape_shape.rank().is_dynamic() || reshape_shape.rank().get_length() != 5 ||
            reshape_shape[2].is_dynamic() || reshape_shape[2].get_length() != 3 ||
            reshape_shape[3].is_dynamic() || reshape_shape[4].is_dynamic()) {
            std::cout << "[QKVSplitReshapeMatcher] FAIL Check2b: shape mismatch\n" << std::flush;
            return false;
        }
        const int64_t H = reshape_shape[3].get_length();
        const int64_t S = reshape_shape[4].get_length();

        // [Check 3] Transpose -> Split(axis=0, num=3)
        if (transpose->get_output_target_inputs(0).size() != 1) {
            std::cout << "[QKVSplitReshapeMatcher] FAIL Check3a: != 1 consumer\n" << std::flush;
            return false;
        }
        auto tp_consumer = (*transpose->get_output_target_inputs(0).begin()).get_node();
        auto split = ov::as_type_ptr<ov::op::v1::Split>(tp_consumer->shared_from_this());
        if (!split || split->get_num_splits() != 3) {
            std::cout << "[QKVSplitReshapeMatcher] FAIL Check3b: no Split or wrong num_splits"
                      << " type=" << tp_consumer->get_type_name() << "\n" << std::flush;
            return false;
        }
        auto split_axis_const = ov::as_type_ptr<ov::op::v0::Constant>(
            split->get_input_node_shared_ptr(1));
        if (!split_axis_const) {
            std::cout << "[QKVSplitReshapeMatcher] FAIL Check3c: no split_axis_const\n" << std::flush;
            return false;
        }
        const auto split_axis_vec = split_axis_const->cast_vector<int64_t>();
        if (split_axis_vec.empty() || split_axis_vec[0] != 0) {
            std::cout << "[QKVSplitReshapeMatcher] FAIL Check3d: wrong split axis=" << (split_axis_vec.empty() ? -99 : split_axis_vec[0]) << "\n" << std::flush;
            return false;
        }
        std::cout << "[QKVSplitReshapeMatcher] Check3 PASS\n" << std::flush;

        // [Check 4 - relaxed] Each Split output -> Squeeze/Reshape (rank 5->4, dim[0]==1 static).
        // In STATIC compile v0::Squeeze is pre-lowered to v1::Reshape; in DYNAMIC it stays v0::Squeeze.
        // Downstream can be:
        //   a) Transpose([0,2,1,3]) -> Reshape  (Q/K path)
        //   b) Anything else, e.g. SDPA         (V path)
        std::array<std::shared_ptr<ov::Node>, 3>              sq_reshapes;      // v1::Reshape OR v0::Squeeze
        std::array<std::shared_ptr<ov::op::v1::Transpose>, 3> qkv_transposes;   // nullptr = V path

        for (size_t i = 0; i < 3; ++i) {
            if (split->get_output_target_inputs(i).size() != 1) {
                std::cout << "[QKVSplitReshapeMatcher] FAIL Check4a[" << i << "]: != 1 consumer\n" << std::flush;
                return false;
            }
            auto sq_node_raw = (*split->get_output_target_inputs(i).begin()).get_node();
            auto sq_node = sq_node_raw->shared_from_this();
            // Accept v1::Reshape (static-lowered Squeeze) OR v0::Squeeze (dynamic path)
            const bool is_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(sq_node) != nullptr;
            const bool is_squeeze = ov::as_type_ptr<ov::op::v0::Squeeze>(sq_node) != nullptr;
            if (!is_reshape && !is_squeeze) {
                std::cout << "[QKVSplitReshapeMatcher] FAIL Check4b[" << i << "]: type=" << sq_node_raw->get_type_name() << "\n" << std::flush;
                return false;
            }
            const auto& sq_in  = split->get_output_partial_shape(i);
            const auto& sq_out = sq_node->get_output_partial_shape(0);
            std::cout << "[QKVSplitReshapeMatcher] Check4[" << i << "] sq_in=" << sq_in << " sq_out=" << sq_out << "\n" << std::flush;
            if (sq_in.rank().is_dynamic()  || sq_in.rank().get_length()  != 5 ||
                sq_out.rank().is_dynamic() || sq_out.rank().get_length() != 4 ||
                sq_in[0].is_dynamic()      || sq_in[0].get_length()      != 1) {
                std::cout << "[QKVSplitReshapeMatcher] FAIL Check4c[" << i << "]\n" << std::flush;
                return false;
            }
            sq_reshapes[i] = sq_node;

            // Check if downstream is Transpose([0,2,1,3]) (Q/K path)
            qkv_transposes[i] = nullptr;
            if (sq_node->get_output_target_inputs(0).size() == 1) {
                auto sq_out_node = (*sq_node->get_output_target_inputs(0).begin()).get_node();
                auto qkv_tp = ov::as_type_ptr<ov::op::v1::Transpose>(sq_out_node->shared_from_this());
                if (qkv_tp) {
                    auto tp_ord_const = ov::as_type_ptr<ov::op::v0::Constant>(
                        qkv_tp->get_input_node_shared_ptr(1));
                    if (tp_ord_const) {
                        const auto tp_ord = tp_ord_const->cast_vector<int64_t>();
                        if (tp_ord == std::vector<int64_t>{0, 2, 1, 3})
                            qkv_transposes[i] = qkv_tp;
                    }
                }
            }
        }

        // === All checks passed — transform ===
        std::cout << "[QKVSplitReshapeMatcher] MATCH at " << transpose->get_friendly_name()
                  << "  H=" << H << " S=" << S << "\n" << std::flush;

        // New Split(axis=2): [?,?,3,H,S] -> [?,?,1,H,S] x3
        auto new_split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2LL});
        auto new_split = std::make_shared<ov::op::v1::Split>(reshape->output(0), new_split_axis, 3);
        new_split->set_friendly_name(split->get_friendly_name());
        ov::copy_runtime_info(split, new_split);

        for (size_t i = 0; i < 3; ++i) {
            // New Squeeze(axis=2): [?,?,1,H,S] -> [?,?,H,S]
            // GPU: crop_axis=2, output_pattern={2}, tentative_reshape_axis=1>=0 -> ALLOW in-place
            auto new_sq_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2LL});
            auto new_squeeze = std::make_shared<ov::op::v0::Squeeze>(new_split->output(i), new_sq_axes);
            new_squeeze->set_friendly_name(sq_reshapes[i]->get_friendly_name());
            ov::copy_runtime_info(sq_reshapes[i], new_squeeze);

            if (qkv_transposes[i]) {
                // Q/K path: new_squeeze output [?,?,H,S] == old qkv_tp output [?,?,H,S].
                // Use replace_node to swap qkv_tp -> new_squeeze in one atomic step.
                ov::replace_node(qkv_transposes[i], new_squeeze);
                std::cout << "[QKVSplitReshapeMatcher] [" << i << "] Q/K: Squeeze(axis=2) "
                          << "replaces Transpose([0,2,1,3])\n" << std::flush;
            } else {
                // V path: consumers expect old sq_reshape output [?,H,?,S].
                // new_squeeze outputs [?,?,H,S]. Add compensating Transpose([0,2,1,3]).
                auto v_tp_order = ov::op::v0::Constant::create(
                    ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 1, 3});
                auto new_v_tp = std::make_shared<ov::op::v1::Transpose>(
                    new_squeeze->output(0), v_tp_order);
                new_v_tp->set_friendly_name(sq_reshapes[i]->get_friendly_name() + "/tp_v");
                ov::copy_runtime_info(sq_reshapes[i], new_v_tp);
                // replace_node atomically replaces all consumers of sq_reshape[i] with new_v_tp
                ov::replace_node(sq_reshapes[i], new_v_tp);
                std::cout << "[QKVSplitReshapeMatcher] [" << i << "] V: Squeeze(axis=2) -> "
                          << "Transpose([0,2,1,3]): replaces old sq_reshape\n" << std::flush;
            }
        }

        std::cout << "[QKVSplitReshapeMatcher] Done\n" << std::flush;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_m, "QKVSplitReshapeMatcher");
    this->register_matcher(m, callback);
}


}  // namespace ov::intel_gpu
