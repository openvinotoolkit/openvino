// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/sdpa_fusion.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"

namespace {

std::vector<size_t> get_order(const ov::pass::pattern::PatternSymbolValue& any_layout_sym,
                              const std::vector<ov::pass::pattern::PatternSymbolValue>& to_find) {
    std::vector<size_t> order;
    const auto& layout = any_layout_sym.g();
    std::set<size_t> already_matched;

    for (const auto& target_sym : to_find) {
        bool found = false;
        for (size_t i = 0; i < layout.size(); ++i) {
            if (layout[i] == target_sym && already_matched.find(i) == already_matched.end()) {
                already_matched.insert(i);
                order.push_back(i);
                found = true;
                ++i;
                break;
            }
        }
        if (!found) {
            return {};  // not all symbols present
        }
    }
    return order;
}

}  // namespace

namespace ov {
namespace pass {

bool SDPAFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(SDPAFusion);
    ov::pass::SymbolicOptimizations symbolic_optimizations(false, get_pass_config());

    auto symbolic_ctx_manager = symbolic_optimizations.get_manager();
    symbolic_ctx_manager->register_pass<ov::pass::SDPAFusionMatcher>();
    symbolic_ctx_manager->register_pass<ov::pass::SDPAReshapeFusion>();
    return symbolic_optimizations.run_on_model(model);
}

SDPAReshapeFusion::SDPAReshapeFusion() {
    MATCHER_SCOPE(SDPAReshapeFusion);

    using namespace ov::op;
    using namespace ov::pass::pattern;

    auto q = any_input(shape_matches("Batches..., S_q, D"));
    auto k = any_input(shape_matches("AnyLayout..."));
    auto v = any_input(shape_matches("Batches..., S_kv, D"));

    // these Reshape/Unsqueeze may already exist in the graph
    auto unsq_q = wrap_type<v1::Reshape, v0::Unsqueeze>({q, any_input()});
    auto unsq_k = wrap_type<v1::Reshape, v0::Unsqueeze>({k, any_input()});
    auto unsq_v = wrap_type<v1::Reshape, v0::Unsqueeze>({v, any_input()});

    // this Transpose may already exist in the graph
    auto opt_original_transpose_k = optional<v1::Transpose>({unsq_k, any_input()});

    // these Reshape/Unsqueeze may be inserted by SDPAFusionMatcher
    auto opt_unsq_q = optional<v1::Reshape, v0::Unsqueeze>({unsq_q, any_input()});
    auto opt_unsq_k = optional<v1::Reshape, v0::Unsqueeze>({opt_original_transpose_k, any_input()});
    auto opt_unsq_v = optional<v1::Reshape, v0::Unsqueeze>({unsq_v, any_input()});

    // this Transpose may be inserted by SDPAFusionMatcher
    auto opt_transpose_k = optional<v1::Transpose>({opt_unsq_k, any_input()}, shape_matches("..., S_kv, D"));

    auto sdpa = wrap_type<v13::ScaledDotProductAttention>({
        opt_unsq_q,
        opt_transpose_k,
        opt_unsq_v,
        any_input(),
        any_input(),
    });

    auto post_sdpa = wrap_type<v1::Reshape, v0::Unsqueeze>({sdpa, any_input()}, shape_matches("Batches..., S_q, D"));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();

        auto q_node = pm.at(q).get_node_shared_ptr();
        auto k_node = pm.at(k).get_node_shared_ptr();
        auto v_node = pm.at(v).get_node_shared_ptr();
        auto sdpa_node = pm.at(sdpa).get_node_shared_ptr();
        auto post_sdpa_node = pm.at(post_sdpa).get_node_shared_ptr();

        if (pm.count(opt_transpose_k)) {
            auto transpose = pm.at(opt_transpose_k).get_node_shared_ptr();
            const auto& sm = m.get_symbols();

            auto symbols_to_find = sm.at("Batches").g();
            auto d_sym = sm.at("D");
            auto s_kv_sym = sm.at("S_kv");
            symbols_to_find.push_back(s_kv_sym);
            symbols_to_find.push_back(d_sym);

            auto any_layout_sym = sm.at("AnyLayout");

            // checks that AnyLayout contains everything from Batches, S_kv and D and returns order
            auto order = get_order(any_layout_sym, symbols_to_find);
            if (order.empty()) {
                k_node = transpose;
            } else {
                size_t idx = 0;
                auto is_identity_order = std::all_of(order.begin(), order.end(), [&idx](size_t x) {
                    return x == idx++;
                });
                if (!is_identity_order) {
                    auto transpose_order = v0::Constant::create(ov::element::i64, {order.size()}, order);
                    k_node = std::make_shared<ov::op::v1::Transpose>(k_node, transpose_order);
                    ov::copy_runtime_info(m.get_matched_nodes(), {transpose_order, k_node});
                }
            }
        }
        auto new_sdpa_node = sdpa_node->clone_with_new_inputs(
            {q_node, k_node, v_node, sdpa_node->input(3).get_source_output(), sdpa_node->input(4).get_source_output()});
        new_sdpa_node->set_friendly_name(post_sdpa_node->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), new_sdpa_node);
        ov::replace_node(post_sdpa_node, new_sdpa_node);
        return true;
    };

    auto m = std::make_shared<Matcher>(post_sdpa, "SDPAReshapeFusion");
    this->register_matcher(m, callback);
}

SDPAFusionMatcher::SDPAFusionMatcher() {
    MATCHER_SCOPE(SDPAFusionMatcher);

    using namespace ov::op;
    using namespace ov::pass::pattern;

    /*
     * Corner Case: Dynamic Mask and Attention Scores
     * When the mask and the attention scores (after MatMul) have [..., -1, -1] shapes,
     * we cannot automatically determine symbols and propogate them.
     * To support pattern matching and avoid false negatives, we treat such cases as if these axes are
     * equal—binding their symbolic names together in the symbol engine accroding to the SDPA specification.
     */
    auto corner_case_check = [](const Output<Node>& out) {
        auto add = out.get_node_shared_ptr();
        auto in0_pshape = add->input(0).get_partial_shape();
        auto out_pshape = add->output(0).get_partial_shape();

        auto corner_case = in0_pshape.rank().get_length() > 1 && out_pshape.rank().get_length() > 1 &&
                           in0_pshape[-1].is_dynamic() && in0_pshape[-2].is_dynamic() && out_pshape[-1].is_dynamic() &&
                           out_pshape[-2].is_dynamic();

        if (corner_case) {
            ov::symbol::set_equal(in0_pshape[-1].get_symbol(), out_pshape[-1].get_symbol());
            ov::symbol::set_equal(in0_pshape[-2].get_symbol(), out_pshape[-2].get_symbol());
        }
        return true;
    };

    auto q = any_input(shape_matches("..., H, S_q, D") || shape_matches("S_q, D"));
    auto k = any_input(shape_matches("..., H, S_kv, D") || shape_matches("S_kv, D"));
    auto kT = any_input(shape_matches("..., H, D, S_kv") || shape_matches("D, S_kv"));
    auto v = any_input(shape_matches("..., H, S_kv, D") || shape_matches("S_kv, D"));

    auto attn_scale = any_input();

    auto opt_k_scale = optional<v1::Multiply>({k, attn_scale});
    auto opt_kT_scale = optional<v1::Multiply>({kT, attn_scale});

    auto qk_pred = (shape_matches("..., H, S_q, S_kv") || shape_matches("S_q, S_kv")) && consumers_count(1);
    auto qk = wrap_type<v0::MatMul>({q, opt_kT_scale}, qk_pred, {{"transpose_a", false}, {"transpose_b", false}});
    auto qk_transpose_b =
        wrap_type<v0::MatMul>({q, opt_k_scale}, qk_pred, {{"transpose_a", false}, {"transpose_b", true}});
    auto qk_alternatives = qk | qk_transpose_b;

    // Optional unsqueeze that is converted to Reshape
    auto unsqueeze_axis = wrap_type<v0::Constant>();
    auto qk_opt_unsqueeze = optional<v1::Reshape>({qk_alternatives, unsqueeze_axis});

    auto qk_scaled = wrap_type<v1::Multiply>({qk_opt_unsqueeze, attn_scale});
    auto qk_opt_scaled = qk_scaled | qk_opt_unsqueeze;

    // Optional nodes:
    // 1. Mask add, there are patterns where before or/and after mask add buffer is reshaped
    // 2. Reshape before adding mask
    // 3. Mask add
    // 4. Reshape after adding mask
    auto mask = any_input(has_static_rank());
    auto qk_opt_scaled_pre_mask_opt_reshaped = optional<v1::Reshape>({qk_opt_scaled, any_input()});

    auto add_pred = consumers_count(1) && corner_case_check;
    auto qk_opt_scaled_opt_mask_added = optional<v1::Add>({qk_opt_scaled_pre_mask_opt_reshaped, mask}, add_pred);
    auto qk_post_mask_opt_reshaped = optional<v1::Reshape>({qk_opt_scaled_opt_mask_added, any_input()});

    auto softmax_pred = (shape_matches("..., H, S_q, S_kv") || shape_matches("S_q, S_kv")) && consumers_count(1);
    auto softmax = wrap_type<v8::Softmax>({qk_post_mask_opt_reshaped}, softmax_pred, {{"axis", -1}});
    auto softmax_opt_reshaped = optional<v1::Reshape>({softmax, any_input()});

    auto qkv_shape = shape_matches("..., H, S_q, D") || shape_matches("S_q, D");
    auto qkv =
        wrap_type<v0::MatMul>({softmax_opt_reshaped, v}, qkv_shape, {{"transpose_a", false}, {"transpose_b", false}});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        // 2 scales detected
        if (pm.count(qk_scaled) && (pm.count(opt_k_scale) || pm.count(opt_kT_scale))) {
            return false;
        }

        bool mask_present = pm.count(mask);
        bool matmul_trasposes_k = pm.count(qk_transpose_b);

        auto q_node = pm.at(q).get_node_shared_ptr();
        auto k_node = pm.count(k) ? pm.at(k).get_node_shared_ptr() : pm.at(kT).get_node_shared_ptr();
        auto v_node = pm.at(v).get_node_shared_ptr();

        if (mask_present && pm.at(mask).get_partial_shape().size() > 4) {
            return false;
        }

        auto T = q_node->output(0).get_element_type();
        ov::Output<ov::Node> scale_node;
        if (pm.count(attn_scale)) {
            scale_node = pm.at(attn_scale);

            // According to the spec, scale should be a scalar or 1D with 1 element
            auto pshape = scale_node.get_partial_shape();
            auto rank = pshape.rank();
            if (rank.is_dynamic()) {
                return false;
            }

            if (pshape.is_static() && ov::shape_size(pshape.get_shape()) != 1) {
                return false;
            } else {
                if (rank.get_length() > 1) {
                    scale_node =
                        ov::op::util::make_try_fold<v1::Reshape>(scale_node,
                                                                 v0::Constant::create(ov::element::i64, {1}, {1}),
                                                                 false);
                }
            }
        } else {
            scale_node = v0::Constant::create(T, ov::Shape{}, {1.0});
        }
        Output<ov::Node> mask_input;
        if (mask_present && pm.count(qk_opt_scaled_opt_mask_added)) {
            ov::Output<ov::Node> qk_out = pm.at(qk_opt_scaled_opt_mask_added);
            // Get shape of the first input
            auto qk_out_ps = qk_out.get_target_inputs().begin()->get_partial_shape();

            mask_input = pm.at(mask);
            auto mask_input_ps = mask_input.get_partial_shape();

            if (!qk_out_ps.rank().is_static() || !mask_input_ps.rank().is_static())
                return false;
            if (qk_out_ps.size() > 4)
                return false;

            std::shared_ptr<v0::Unsqueeze> mask_unsqueeze;
            // mask should be broadcastable to qk shape
            if (!ov::PartialShape::broadcast_merge_into(qk_out_ps, mask_input_ps, AutoBroadcastType::NUMPY))
                return false;

            if (mask_input_ps.size() < qk_out_ps.size()) {
                size_t rank_diff = qk_out_ps.size() - mask_input_ps.size();
                std::vector<int64_t> axes(rank_diff);
                std::iota(axes.begin(), axes.end(), 0);
                mask_unsqueeze =
                    std::make_shared<v0::Unsqueeze>(mask_input,
                                                    v0::Constant::create(ov::element::i64, ov::Shape{rank_diff}, axes));
                mask_unsqueeze->set_friendly_name(mask->get_friendly_name());
                ov::copy_runtime_info(m.get_matched_nodes(), mask_unsqueeze);
                mask_input = mask_unsqueeze;
            }
        } else {
            mask_input = v0::Constant::create(T, ov::Shape{}, {0});
        }

        ov::OutputVector vec = {q_node, k_node, v_node};
        int supported_rank = 3;  // this is the min supported rank according to the SDPA spec
        for (size_t i = 0; i < vec.size(); ++i) {
            auto pshape = vec[i].get_partial_shape();
            if (pshape.rank().is_dynamic()) {
                return false;
            }
            // align all inputs
            supported_rank = std::max(static_cast<int>(pshape.size()), supported_rank);
        }

        for (size_t i = 0; i < vec.size(); ++i) {
            auto pshape = vec[i].get_partial_shape();
            int diff = supported_rank - static_cast<int>(pshape.size());
            if (diff > 0) {
                std::vector<size_t> axes(diff, 0);
                std::iota(axes.begin(), axes.end(), 0);
                auto axes_node = v0::Constant::create(ov::element::i64, ov::Shape{static_cast<size_t>(diff)}, axes);
                auto reshape = std::make_shared<v0::Unsqueeze>(vec[i], axes_node);
                vec[i] = reshape;
                ov::copy_runtime_info(m.get_matched_nodes(), {reshape, axes_node});
            }

            if (i == 1 && !matmul_trasposes_k) {
                // Transpose k
                pshape = vec[i].get_partial_shape();
                std::vector<int> axes_values(pshape.size());
                std::iota(axes_values.begin(), axes_values.end(), 0);
                std::swap(axes_values[axes_values.size() - 1], axes_values[axes_values.size() - 2]);
                auto axes = v0::Constant::create(ov::element::i64, {axes_values.size()}, axes_values);
                vec[i] = std::make_shared<v1::Transpose>(vec[i], axes);
                ov::copy_runtime_info(m.get_matched_nodes(), {axes, vec[i].get_node_shared_ptr()});
            }
        }

        std::shared_ptr<ov::Node> sdpa =
            std::make_shared<v13::ScaledDotProductAttention>(vec[0], vec[1], vec[2], mask_input, scale_node, false);
        sdpa->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);
        ov::replace_node(m.get_match_root(), sdpa);
        return true;
    };

    auto m = std::make_shared<Matcher>(qkv, "SDPAFusionMatcher");
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace ov
