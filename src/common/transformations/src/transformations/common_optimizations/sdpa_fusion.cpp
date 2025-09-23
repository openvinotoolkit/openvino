// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/sdpa_fusion.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
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

    // the ranks have to be equal
    if (layout.size() != to_find.size()) {
        return {};
    }

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

ov::pass::pattern::op::Predicate check_layout(const std::string& layout) {
    return ov::pass::pattern::op::Predicate(
        [=](ov::pass::pattern::PatternSymbolMap& sm, const ov::Output<ov::Node>& output) -> bool {
            if (!sm.count("D") || !sm.count("S_kv") || !sm.count("Batches") || !sm.count("AnyLayout")) {
                return false;
            }

            // checks that AnyLayout contains everything from Batches, S_kv and D and returns order
            auto symbols_to_find = sm.at("Batches").g();
            auto d_sym = sm.at("D");
            auto s_kv_sym = sm.at("S_kv");
            symbols_to_find.push_back(s_kv_sym);
            symbols_to_find.push_back(d_sym);

            auto any_layout_sym = sm.at("AnyLayout");
            auto order = get_order(any_layout_sym, symbols_to_find);

            return !order.empty();
        });
};
}  // namespace

namespace ov {
namespace pass {

bool SDPAFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(SDPAFusion);
    ov::pass::SymbolicOptimizations symbolic_optimizations(false, get_pass_config());
    auto symbolic_ctx_manager = symbolic_optimizations.get_manager();
    symbolic_ctx_manager->register_pass<ov::pass::SDPAFusionMatcher>();
    symbolic_ctx_manager->register_pass<ov::pass::SDPAFusionMatcherSinks>();
    symbolic_ctx_manager->register_pass<ov::pass::SDPAReshapeFusion>();
    return symbolic_optimizations.run_on_model(model);
}

SDPAReshapeFusion::SDPAReshapeFusion() {
    MATCHER_SCOPE(SDPAReshapeFusion);

    using namespace ov::op;
    using namespace ov::pass::pattern;

    auto q = any_input(shape_matches("Batches..., S_q, D") && rank_more_than(2));
    auto k = any_input(shape_matches("AnyLayout...") && rank_more_than(2));
    auto v = any_input(shape_matches("Batches..., S_kv, D") && check_layout("AnyLayout") && rank_more_than(2));

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
    auto opt_transpose_k =
        optional<v1::Transpose>({opt_unsq_k, any_input()}, shape_matches("..., S_kv, D") && rank_more_than(2));

    auto sdpa = wrap_type<v13::ScaledDotProductAttention>({
        opt_unsq_q,
        opt_transpose_k,
        opt_unsq_v,
        any_input(),
        any_input(),
    });

    auto opt_sdpa_reshape = optional<v1::Reshape, v0::Unsqueeze>({sdpa->output(0), any_input()});
    auto opt_sdpa_transpose = optional<v1::Transpose>({opt_sdpa_reshape, any_input()});
    auto post_sdpa = wrap_type<v1::Reshape, v0::Unsqueeze>({opt_sdpa_transpose, any_input()},
                                                           shape_matches("Batches..., S_q, D") && rank_more_than(2));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();

        auto q_node = pm.at(q);
        auto k_node = pm.at(k);
        auto v_node = pm.at(v);
        auto sdpa_node = pm.at(sdpa).get_node_shared_ptr();
        auto post_sdpa_node = pm.at(post_sdpa).get_node_shared_ptr();

        const auto& sm = m.get_symbols();

        auto symbols_to_find = sm.at("Batches").g();
        auto d_sym = sm.at("D");
        auto s_kv_sym = sm.at("S_kv");
        symbols_to_find.push_back(s_kv_sym);
        symbols_to_find.push_back(d_sym);

        auto any_layout_sym = sm.at("AnyLayout");

        // checks that AnyLayout contains everything from Batches, S_kv and D and returns order
        auto order = get_order(any_layout_sym, symbols_to_find);
        size_t idx = 0;
        auto is_ascending_order = std::all_of(order.begin(), order.end(), [&idx](size_t x) {
            return x == idx++;
        });
        if (pm.count(opt_transpose_k)) {
            auto transpose = pm.at(opt_transpose_k).get_node_shared_ptr();
            if (order.empty()) {
                k_node = transpose;
            } else {
                if (!is_ascending_order) {
                    auto transpose_order = v0::Constant::create(ov::element::i64, {order.size()}, order);
                    k_node = std::make_shared<ov::op::v1::Transpose>(k_node, transpose_order);
                    ov::copy_runtime_info(m.get_matched_nodes(), {transpose_order, k_node.get_node_shared_ptr()});
                }
            }
        } else {
            if (!is_ascending_order) {
                auto shape_of_v = ov::op::util::make_try_fold<ov::op::v3::ShapeOf>(v_node);
                k_node = std::make_shared<ov::op::v1::Reshape>(k_node, shape_of_v, false);
                ov::copy_runtime_info(m.get_matched_nodes(), {shape_of_v, k_node.get_node_shared_ptr()});
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

    auto q = any_input(shape_matches("..., H, S_q, E") || shape_matches("S_q, E"));
    auto k = any_input(shape_matches("..., H, S_kv, E") || shape_matches("S_kv, E"));
    auto kT = any_input(shape_matches("..., H, E, S_kv") || shape_matches("E, S_kv"));
    auto v = any_input(shape_matches("..., H, S_kv, Ev") || shape_matches("S_kv, Ev"));

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

    // Softmax axis can be:
    // Pattern 1: axis = -1 (last axis)
    // Pattern 2: axis = rank size - 1 (also means last axis for static rank inputs)
    auto axis_predicate = ([](const ov::Output<ov::Node>& node) {
        auto softmax = std::dynamic_pointer_cast<ov::op::v8::Softmax>(node.get_node_shared_ptr());
        if (!softmax)
            return false;
        auto input_rank = node.get_partial_shape().rank();
        if (input_rank.is_dynamic())
            return false;
        auto axis = ov::util::try_normalize_axis(softmax->get_axis(), input_rank, *softmax);
        return static_cast<size_t>(input_rank.get_length() - 1) == axis;
    });
    auto softmax_pred =
        consumers_count(1) && axis_predicate && (shape_matches("..., H, S_q, S_kv") || shape_matches("S_q, S_kv"));
    auto softmax = wrap_type<v8::Softmax>({qk_post_mask_opt_reshaped}, softmax_pred);
    auto softmax_opt_reshaped = optional<v1::Reshape>({softmax, any_input()});

    auto qkv_shape = shape_matches("..., H, S_q, Ev") || shape_matches("S_q, Ev");
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

        auto q_node = pm.at(q);
        auto k_node = pm.count(k) ? pm.at(k) : pm.at(kT);
        auto v_node = pm.at(v);

        if (mask_present && pm.at(mask).get_partial_shape().size() > 4) {
            return false;
        }

        auto T = q_node.get_element_type();
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

            // mask should be broadcastable to qk shape
            if (!ov::PartialShape::broadcast_merge_into(qk_out_ps, mask_input_ps, AutoBroadcastType::NUMPY))
                return false;

            if (mask_input_ps.size() < 2) {
                // OpenVINO SDPA specification requires the attention mask to have rank >= 2.
                auto diff = 2 - mask_input_ps.size();
                std::vector<int64_t> axes(diff);
                std::iota(axes.begin(), axes.end(), 0);
                auto axes_const = v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
                auto mask_unsqueeze = std::make_shared<v0::Unsqueeze>(mask_input, axes_const);
                mask_unsqueeze->set_friendly_name(mask->get_friendly_name());
                ov::copy_runtime_info(m.get_matched_nodes(), mask_unsqueeze);
                mask_input = mask_unsqueeze;
            } else {
                std::vector<int64_t> axes;
                // -2 because OpenVINO SDPA specification requires the attention mask to have rank >= 2.
                for (size_t i = 0; i < (mask_input_ps.size() - 2); ++i) {
                    if (mask_input_ps[i].is_static() && mask_input_ps[i].get_length() == 1) {
                        axes.push_back(i);
                    } else {
                        break;
                    }
                }
                if (!axes.empty()) {
                    auto axes_const = v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
                    auto mask_squeeze = std::make_shared<v0::Squeeze>(mask_input, axes_const);
                    mask_squeeze->set_friendly_name(mask->get_friendly_name());
                    ov::copy_runtime_info(m.get_matched_nodes(), mask_squeeze);
                    mask_input = mask_squeeze;
                }
            }
        } else {
            mask_input = v0::Constant::create(T, ov::Shape{}, {0});
        }

        ov::OutputVector vec = {q_node, k_node, v_node};
        // 3 is the min supported rank according to the SDPA spec
        int64_t supported_rank = std::max(mask_input.get_partial_shape().rank().get_length(), static_cast<int64_t>(3));
        for (size_t i = 0; i < vec.size(); ++i) {
            auto pshape = vec[i].get_partial_shape();
            if (pshape.rank().is_dynamic()) {
                return false;
            }
            // align all inputs
            supported_rank = std::max(static_cast<int64_t>(pshape.size()), supported_rank);
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

SDPAFusionMatcherSinks::SDPAFusionMatcherSinks() {
    MATCHER_SCOPE(SDPAFusionMatcherSinks);

    using namespace ov::op;
    using namespace ov::pass::pattern;

    auto v = any_input(rank_equals(4));
    auto k = any_input(rank_equals(4));
    auto q = any_input(rank_equals(4));

    auto qk_transpose_b =
        wrap_type<v0::MatMul>({q, k}, consumers_count(1), {{"transpose_a", false}, {"transpose_b", true}});

    auto attn_scale = any_input();
    auto opt_qk_scaled = optional<v1::Multiply>({qk_transpose_b, attn_scale});

    auto mask = any_input(has_static_rank());
    auto add_pred = consumers_count(1);
    auto opt_mask_add = optional<v1::Add>({opt_qk_scaled, mask}, add_pred);

    auto sinks = any_input();
    auto sinks_broadcast = wrap_type<v3::Broadcast>({sinks, any_input()});
    auto sinks_concat = wrap_type<v0::Concat>({opt_mask_add, sinks_broadcast});
    auto sinks_rm = wrap_type<v1::ReduceMax>({sinks_concat, any_input()});
    auto sinks_sub = wrap_type<v1::Subtract>({sinks_concat, sinks_rm});

    // Softmax axis can be:
    // Pattern 1: axis = -1 (last axis)
    // Pattern 2: axis = rank size - 1 (also means last axis for static rank inputs)
    auto axis_predicate = ([](const ov::Output<ov::Node>& node) {
        auto softmax = ov::as_type_ptr<ov::op::v8::Softmax>(node.get_node_shared_ptr());
        if (!softmax)
            return false;
        auto input_rank = node.get_partial_shape().rank();
        if (input_rank.is_dynamic())
            return false;
        auto axis = ov::util::try_normalize_axis(softmax->get_axis(), input_rank, *softmax);
        return static_cast<size_t>(input_rank.get_length() - 1) == axis;
    });
    auto softmax_pred = consumers_count(1) && axis_predicate;
    auto softmax = wrap_type<v8::Softmax>({sinks_sub}, softmax_pred);

    auto sinks_ss = wrap_type<v1::StridedSlice>({softmax, any_input(), any_input(), any_input()});
    auto qkv = wrap_type<v0::MatMul>({sinks_ss, v});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        bool mask_present = pm.count(mask);
        bool matmul_trasposes_k = pm.count(qk_transpose_b);

        auto q_node = pm.at(q);
        auto k_node = pm.at(k);
        auto v_node = pm.at(v);

        if (pm.at(mask).get_partial_shape().rank().get_length() > 4) {
            return false;
        }

        auto T = q_node.get_element_type();
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
        if (mask_present && pm.count(opt_mask_add)) {
            ov::Output<ov::Node> qk_out = pm.at(opt_mask_add);
            // Get shape of the first input
            auto qk_out_ps = qk_out.get_target_inputs().begin()->get_partial_shape();

            mask_input = pm.at(mask);
            auto mask_input_ps = mask_input.get_partial_shape();

            if (!qk_out_ps.rank().is_static() || !mask_input_ps.rank().is_static())
                return false;
            if (qk_out_ps.size() > 4)
                return false;

            // mask should be broadcastable to qk shape
            if (!ov::PartialShape::broadcast_merge_into(qk_out_ps, mask_input_ps, AutoBroadcastType::NUMPY))
                return false;

            if (mask_input_ps.size() < 2) {
                // OpenVINO SDPA specification requires the attention mask to have rank >= 2.
                auto diff = 2 - mask_input_ps.size();
                std::vector<int64_t> axes(diff);
                std::iota(axes.begin(), axes.end(), 0);
                auto axes_const = v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
                auto mask_unsqueeze = std::make_shared<v0::Unsqueeze>(mask_input, axes_const);
                mask_unsqueeze->set_friendly_name(mask->get_friendly_name());
                ov::copy_runtime_info(m.get_matched_nodes(), mask_unsqueeze);
                mask_input = mask_unsqueeze;
            } else {
                std::vector<int64_t> axes;
                // -2 because OpenVINO SDPA specification requires the attention mask to have rank >= 2.
                for (size_t i = 0; i < (mask_input_ps.size() - 2); ++i) {
                    if (mask_input_ps[i].is_static() && mask_input_ps[i].get_length() == 1) {
                        axes.push_back(i);
                    } else {
                        break;
                    }
                }
                if (!axes.empty()) {
                    auto axes_const = v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
                    auto mask_squeeze = std::make_shared<v0::Squeeze>(mask_input, axes_const);
                    mask_squeeze->set_friendly_name(mask->get_friendly_name());
                    ov::copy_runtime_info(m.get_matched_nodes(), mask_squeeze);
                    mask_input = mask_squeeze;
                }
            }
        } else {
            mask_input = v0::Constant::create(T, ov::Shape{}, {0});
        }

        ov::OutputVector vec = {q_node, k_node, v_node};
        // 3 is the min supported rank according to the SDPA spec
        int64_t supported_rank = std::max(mask_input.get_partial_shape().rank().get_length(), static_cast<int64_t>(3));
        for (size_t i = 0; i < vec.size(); ++i) {
            auto pshape = vec[i].get_partial_shape();
            if (pshape.rank().is_dynamic()) {
                return false;
            }
            // align all inputs
            supported_rank = std::max(static_cast<int64_t>(pshape.size()), supported_rank);
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

        bool sinks_present = pm.count(sinks);
        std::shared_ptr<ov::Node> sdpa = sinks_present ? std::make_shared<v13::ScaledDotProductAttention>(vec[0],
                                                                                                          vec[1],
                                                                                                          vec[2],
                                                                                                          mask_input,
                                                                                                          scale_node,
                                                                                                          pm.at(sinks),
                                                                                                          false)
                                                       : std::make_shared<v13::ScaledDotProductAttention>(vec[0],
                                                                                                          vec[1],
                                                                                                          vec[2],
                                                                                                          mask_input,
                                                                                                          scale_node,
                                                                                                          false);
        sdpa->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);
        ov::replace_node(m.get_match_root(), sdpa);
        return true;
    };

    auto m = std::make_shared<Matcher>(qkv, "SDPAFusionMatcherSinks");
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace ov
