// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sequence_array_lowering.hpp"

#include <functional>
#include <map>
#include <numeric>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/sequence_at.hpp"
#include "openvino/frontend/sequence_erase.hpp"
#include "openvino/frontend/sequence_insert.hpp"
#include "openvino/frontend/sequence_length.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/identity.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/util/log.hpp"
#include "slot_resolver.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace pass {
namespace sal_detail {

namespace {

bool is_sequence_helper(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type<ov::frontend::SequenceMark>(node) || ov::is_type<ov::frontend::SequenceInsert>(node) ||
           ov::is_type<ov::frontend::SequenceErase>(node) || ov::is_type<ov::frontend::SequenceAt>(node) ||
           ov::is_type<ov::frontend::SequenceLength>(node);
}

void collect_helpers_recursive(const std::shared_ptr<ov::Model>& m, std::vector<std::shared_ptr<ov::Node>>& out) {
    if (!m) {
        return;
    }
    for (const auto& n : m->get_ordered_ops()) {
        if (is_sequence_helper(n)) {
            out.push_back(n);
        }
        if (auto msg = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(n)) {
            for (size_t i = 0; i < msg->get_internal_subgraphs_size(); ++i) {
                collect_helpers_recursive(msg->get_function(i), out);
            }
        }
    }
}

// True if model (recursively) contains SequenceAt or SequenceLength.
bool model_has_sequence_reader(const std::shared_ptr<ov::Model>& m) {
    if (!m) {
        return false;
    }
    for (const auto& n : m->get_ordered_ops()) {
        if (ov::is_type<ov::frontend::SequenceAt>(n) || ov::is_type<ov::frontend::SequenceLength>(n)) {
            return true;
        }
        if (auto msg = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(n)) {
            for (size_t i = 0; i < msg->get_internal_subgraphs_size(); ++i) {
                if (model_has_sequence_reader(msg->get_function(i))) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool produces_sequence_helper_value(const ov::Output<ov::Node>& value) {
    auto u = unwrap_identity(value);
    auto n = u.get_node_shared_ptr();
    return ov::is_type<ov::frontend::SequenceMark>(n) || ov::is_type<ov::frontend::SequenceInsert>(n) ||
           ov::is_type<ov::frontend::SequenceErase>(n);
}

// Apply fn(model) to every model in the subgraph tree, post-order (children before parent).
template <typename Fn>
void for_each_model_postorder(const std::shared_ptr<ov::Model>& m, Fn&& fn) {
    if (!m) {
        return;
    }
    for (const auto& n : m->get_ordered_ops()) {
        if (auto msg = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(n)) {
            for (size_t i = 0; i < msg->get_internal_subgraphs_size(); ++i) {
                for_each_model_postorder(msg->get_function(i), fn);
            }
        }
    }
    fn(m);
}

// Disconnect body Results that still source sequence helpers from back-edges.
void disconnect_dead_sequence_back_edges(const std::shared_ptr<ov::Model>& root) {
    for_each_model_postorder(root, [&](const std::shared_ptr<ov::Model>& m) {
        for (const auto& n : m->get_ordered_ops()) {
            auto msg = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(n);
            if (!msg) {
                continue;
            }
            for (size_t b = 0; b < msg->get_internal_subgraphs_size(); ++b) {
                auto body = msg->get_function(b);
                if (!body) {
                    continue;
                }
                const auto& body_results = body->get_results();
                std::map<size_t, std::shared_ptr<v0::Parameter>> merged_result_to_param;
                for (const auto& d : msg->get_input_descriptions(static_cast<int>(b))) {
                    if (auto merged = ov::as_type_ptr<ov::op::util::MultiSubGraphOp::MergedInputDescription>(d)) {
                        if (merged->m_body_value_index < body_results.size() &&
                            merged->m_body_parameter_index < body->get_parameters().size()) {
                            merged_result_to_param[merged->m_body_value_index] =
                                body->get_parameters()[merged->m_body_parameter_index];
                        }
                    }
                }
                for (size_t r_idx = 0; r_idx < body_results.size(); ++r_idx) {
                    auto& result = body_results[r_idx];
                    auto src = result->input_value(0);
                    if (!produces_sequence_helper_value(src)) {
                        continue;
                    }
                    auto merged_it = merged_result_to_param.find(r_idx);
                    if (merged_it != merged_result_to_param.end()) {
                        // Skip if the merged Parameter still has live (non-Result) consumers.
                        const auto& param_out = merged_it->second->output(0);
                        size_t live_consumers = 0;
                        for (const auto& tgt : param_out.get_target_inputs()) {
                            if (!ov::is_type<v0::Result>(tgt.get_node())) {
                                ++live_consumers;
                            }
                        }
                        if (live_consumers > 0) {
                            continue;  // back-edge is live; leave in place
                        }
                        result->input(0).replace_source_output(param_out);
                        continue;
                    }
                    // For a BodyOutput (non-merged), replace with a shape-compatible
                    // zero seed derived from the sibling branch's result.
                    int64_t out_idx_for_r = -1;
                    for (const auto& d : msg->get_output_descriptions(static_cast<int>(b))) {
                        if (d->m_body_value_index == r_idx) {
                            out_idx_for_r = static_cast<int64_t>(d->m_output_index);
                            break;
                        }
                    }
                    ov::Output<ov::Node> sibling_src;
                    if (out_idx_for_r >= 0) {
                        for (size_t bb = 0; bb < msg->get_internal_subgraphs_size(); ++bb) {
                            if (bb == b) {
                                continue;
                            }
                            auto sib_body = msg->get_function(bb);
                            if (!sib_body) {
                                continue;
                            }
                            for (const auto& d : msg->get_output_descriptions(static_cast<int>(bb))) {
                                if (static_cast<int64_t>(d->m_output_index) != out_idx_for_r) {
                                    continue;
                                }
                                const auto& sib_results = sib_body->get_results();
                                if (d->m_body_value_index >= sib_results.size()) {
                                    continue;
                                }
                                auto candidate = sib_results[d->m_body_value_index]->input_value(0);
                                if (produces_sequence_helper_value(candidate)) {
                                    continue;
                                }
                                sibling_src = candidate;
                                break;
                            }
                            if (sibling_src.get_node_shared_ptr()) {
                                break;
                            }
                        }
                    }
                    ov::Output<ov::Node> dummy;
                    if (sibling_src.get_node_shared_ptr()) {
                        const auto& tps = sibling_src.get_partial_shape();
                        auto et = sibling_src.get_element_type();
                        if (et == ov::element::dynamic) {
                            et = src.get_element_type();
                        }
                        if (et == ov::element::dynamic && out_idx_for_r >= 0) {
                            et = msg->output(static_cast<size_t>(out_idx_for_r)).get_element_type();
                        }
                        if (et == ov::element::dynamic) {
                            et = ov::element::f32;
                        }
                        dummy = make_growable_seed(tps, et);
                    } else {
                        dummy = v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f})->output(0);
                    }
                    result->input(0).replace_source_output(dummy);
                }
            }
        }
    });
}

// Detach dead sequence helpers by replacing their inputs with Constants.
void disconnect_dead_sequence_helpers(const std::shared_ptr<ov::Model>& root) {
    for_each_model_postorder(root, [&](const std::shared_ptr<ov::Model>& m) {
        bool progress = true;
        while (progress) {
            progress = false;
            for (const auto& n : m->get_ordered_ops()) {
                if (!is_sequence_helper(n)) {
                    continue;
                }
                if (!n->output(0).get_target_inputs().empty()) {
                    continue;
                }
                if (n->get_input_size() == 0) {
                    continue;
                }
                for (size_t i = 0; i < n->get_input_size(); ++i) {
                    auto src = n->input_value(i);
                    if (ov::is_type<v0::Constant>(src.get_node_shared_ptr())) {
                        continue;
                    }
                    auto et = src.get_element_type();
                    if (et.is_dynamic()) {
                        et = ov::element::i64;
                    }
                    auto c = v0::Constant::create(et, ov::Shape{}, std::vector<int64_t>{0});
                    n->input(i).replace_source_output(c);
                    progress = true;
                }
            }
        }
    });
}

// Fix If branches where one returns rank-0 and another returns rank-N for the same output.
void align_if_branch_result_ranks(const std::shared_ptr<ov::Model>& root) {
    for_each_model_postorder(root, [&](const std::shared_ptr<ov::Model>& m) {
        for (const auto& n : m->get_ordered_ops()) {
            auto if_op = ov::as_type_ptr<v8::If>(n);
            if (!if_op) {
                continue;
            }
            auto then_body = if_op->get_then_body();
            auto else_body = if_op->get_else_body();
            if (!then_body || !else_body) {
                continue;
            }
            struct PerOut {
                int then_r = -1;
                int else_r = -1;
            };
            std::map<int64_t, PerOut> per_out;
            for (const auto& d : if_op->get_output_descriptions(0)) {
                per_out[static_cast<int64_t>(d->m_output_index)].then_r = static_cast<int>(d->m_body_value_index);
            }
            for (const auto& d : if_op->get_output_descriptions(1)) {
                per_out[static_cast<int64_t>(d->m_output_index)].else_r = static_cast<int>(d->m_body_value_index);
            }
            for (const auto& kv : per_out) {
                int t_r = kv.second.then_r;
                int e_r = kv.second.else_r;
                if (t_r < 0 || e_r < 0) {
                    continue;
                }
                auto t_res = then_body->get_results()[t_r];
                auto e_res = else_body->get_results()[e_r];
                auto t_src = t_res->input_value(0);
                auto e_src = e_res->input_value(0);
                auto t_ps = t_src.get_partial_shape();
                auto e_ps = e_src.get_partial_shape();
                if (t_ps.rank().is_dynamic() || e_ps.rank().is_dynamic()) {
                    continue;
                }
                if (t_ps.rank().get_length() == e_ps.rank().get_length()) {
                    continue;
                }
                bool then_is_smaller = t_ps.rank().get_length() < e_ps.rank().get_length();
                auto& placeholder_res = then_is_smaller ? t_res : e_res;
                auto placeholder_body = then_is_smaller ? then_body : else_body;
                auto template_src = then_is_smaller ? e_src : t_src;
                const auto tmpl_ps = template_src.get_partial_shape();
                const auto tmpl_et = template_src.get_element_type();

                // Hoist the live branch's source into the parent scope by cloning its
                // def-chain, route it as a new If input into the placeholder branch,
                // so the placeholder produces the same runtime value.
                ov::Output<ov::Node> replacement;
                {
                    const bool template_is_then = !then_is_smaller;
                    const int tmpl_branch_idx = template_is_then ? 0 : 1;
                    auto template_body = template_is_then ? then_body : else_body;

                    std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, int)> hoist;
                    hoist = [&](const ov::Output<ov::Node>& v, int depth_left) -> ov::Output<ov::Node> {
                        if (depth_left <= 0) {
                            return {};
                        }
                        auto nd = v.get_node_shared_ptr();
                        if (auto p = ov::as_type_ptr<v0::Parameter>(nd)) {
                            const int64_t param_idx = template_body->get_parameter_index(p);
                            if (param_idx < 0) {
                                return {};
                            }
                            for (const auto& d : if_op->get_input_descriptions(tmpl_branch_idx)) {
                                if (d->m_body_parameter_index == static_cast<size_t>(param_idx)) {
                                    return if_op->input_value(static_cast<int>(d->m_input_index));
                                }
                            }
                            return {};
                        }
                        if (auto c = ov::as_type_ptr<v0::Constant>(nd)) {
                            return c->clone_with_new_inputs({})->output(0);
                        }
                        if (ov::as_type<ov::op::util::MultiSubGraphOp>(nd.get())) {
                            return {};
                        }
                        ov::OutputVector new_inputs;
                        new_inputs.reserve(nd->get_input_size());
                        for (size_t i = 0; i < nd->get_input_size(); ++i) {
                            auto h = hoist(nd->input_value(i), depth_left - 1);
                            if (!h.get_node_shared_ptr()) {
                                return {};
                            }
                            new_inputs.push_back(h);
                        }
                        auto cloned = nd->clone_with_new_inputs(new_inputs);
                        return cloned->output(v.get_index());
                    };

                    auto hoisted = hoist(template_src, /*depth_left=*/32);
                    // Unwrap any cloned Identity to avoid orphaned cross-scope edges.
                    hoisted = unwrap_identity(hoisted);
                    if (hoisted.get_node_shared_ptr()) {
                        auto new_param =
                            std::make_shared<v0::Parameter>(hoisted.get_element_type(), hoisted.get_partial_shape());
                        new_param->set_friendly_name(placeholder_res->get_friendly_name() + "/hoist");
                        placeholder_body->add_parameters({new_param});
                        std::shared_ptr<v0::Parameter> then_p = then_is_smaller ? new_param : nullptr;
                        std::shared_ptr<v0::Parameter> else_p = then_is_smaller ? nullptr : new_param;
                        if_op->set_input(hoisted, then_p, else_p);
                        if_op->input(if_op->get_input_size() - 1).replace_source_output(hoisted);
                        placeholder_res->input(0).replace_source_output(new_param->output(0));
                        continue;
                    }
                }

                // Fallback: use a matching body Parameter as a pass-through
                // when hoisting is not viable.
                for (const auto& p : placeholder_body->get_parameters()) {
                    if (p->get_element_type() != tmpl_et) {
                        continue;
                    }
                    const auto p_ps = p->get_partial_shape();
                    if (p_ps.rank().is_dynamic() || p_ps.rank().get_length() != tmpl_ps.rank().get_length()) {
                        continue;
                    }
                    bool ok = true;
                    for (size_t k = 0; k < p_ps.size(); ++k) {
                        // Reject zero-sized dims (placeholder dummies, not live state).
                        if (p_ps[k].is_static() && p_ps[k].get_length() == 0) {
                            ok = false;
                            break;
                        }
                        if (!p_ps[k].compatible(tmpl_ps[k])) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) {
                        replacement = p->output(0);
                        break;
                    }
                }
                if (!replacement.get_node_shared_ptr()) {
                    // Zero constant matching template shape (dynamic dims -> 1).
                    if (tmpl_et.is_real() || tmpl_et.is_integral()) {
                        ov::Shape placeholder_shape;
                        placeholder_shape.reserve(tmpl_ps.size());
                        for (size_t k = 0; k < tmpl_ps.size(); ++k) {
                            placeholder_shape.push_back(
                                tmpl_ps[k].is_static() ? static_cast<size_t>(tmpl_ps[k].get_length()) : size_t{1});
                        }
                        replacement = v0::Constant::create(tmpl_et, placeholder_shape, {0.0f})->output(0);
                    } else {
                        replacement = make_zero_dummy(template_src);
                    }
                }
                placeholder_res->input(0).replace_source_output(replacement);
            }
        }
    });
}

// Lower SequenceAt to a value picked from the resolved per-slot tensors.
//
//   slots = [s0, s1, s2]
//   At(idx=1)        ==>  s1                                  (static index)
//   At(idx=dynamic)  ==>  If(idx==0, s0, If(idx==1, s1, s2))  (shape-preserving)
bool lower_sequence_at(const std::shared_ptr<ov::frontend::SequenceAt>& at, SlotResolver& resolver) {
    auto slots = resolver.slots_of(at->input_value(0));
    if (!slots || slots->empty()) {
        return false;
    }
    auto idx_c = ov::util::get_constant_from_source(at->input_value(1));
    if (idx_c) {
        const auto iv = idx_c->cast_vector<int64_t>();
        if (iv.size() != 1) {
            return false;
        }
        int64_t k = iv[0];
        if (k < 0) {
            k += static_cast<int64_t>(slots->size());
        }
        if (k < 0 || static_cast<size_t>(k) >= slots->size()) {
            return false;
        }
        at->output(0).replace((*slots)[static_cast<size_t>(k)]);
        return true;
    }
    // Dynamic index: out = slots[N-1]; for j=N-2..0: out = If(idx==j, slots[j], out)
    auto idx_in = at->input_value(1);
    auto idx_i64 = std::make_shared<v0::Convert>(idx_in, ov::element::i64)->output(0);
    if (idx_i64.get_partial_shape().rank().is_static() && idx_i64.get_partial_shape().rank().get_length() > 0) {
        const auto rank = idx_i64.get_partial_shape().rank().get_length();
        std::vector<int64_t> axes_vec(rank);
        std::iota(axes_vec.begin(), axes_vec.end(), int64_t{0});
        auto axes = v0::Constant::create(ov::element::i64, ov::Shape{static_cast<size_t>(rank)}, axes_vec);
        idx_i64 = std::make_shared<v0::Squeeze>(idx_i64, axes)->output(0);
    }
    ov::Output<ov::Node> out = slots->back();
    for (int64_t j = static_cast<int64_t>(slots->size()) - 2; j >= 0; --j) {
        auto j_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {j});
        auto eq = std::make_shared<v1::Equal>(idx_i64, j_const)->output(0);
        out = make_shape_preserving_select(eq, (*slots)[static_cast<size_t>(j)], out);
    }
    at->output(0).replace(out);
    return true;
}

// Lower SequenceLength to the element count of the resolved slots. Returns
// false (leaves the reader as-is) when the accumulation axis is ambiguous.
//
//   slots grow on a dynamic axis  ==>  ShapeOf(s0)[axis]   (runtime length)
//   slots fully static            ==>  Const(N)            (slot count)
bool lower_sequence_length(const std::shared_ptr<ov::frontend::SequenceLength>& len, SlotResolver& resolver) {
    auto slots = resolver.slots_of(len->input_value(0));
    if (!slots) {
        return false;
    }
    // Loop-carried sequence from SequenceEmpty: count is a runtime property.
    // Emit (ReduceProd(shape(s0)) == 0) ? 0 : N rather than static N.
    if (resolver.is_loop_carried_empty_seed(len->input_value(0)) && !slots->empty()) {
        const auto& s0 = (*slots)[0];
        const auto n = static_cast<int64_t>(slots->size());
        auto shape_of = std::make_shared<v3::ShapeOf>(s0, ov::element::i64);
        auto all_axes = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto num_elems = std::make_shared<ov::op::v1::ReduceProd>(shape_of, all_axes, /*keep_dims=*/false);
        auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto is_empty = std::make_shared<v1::Equal>(num_elems, zero);
        auto n_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {n});
        auto count = std::make_shared<v1::Select>(is_empty, zero, n_const)->output(0);
        len->output(0).replace(count);
        return true;
    }
    ov::Output<ov::Node> length_value;
    bool axis_ambiguous = false;
    if (!slots->empty()) {
        const auto& s0 = (*slots)[0];
        const auto& ps = s0.get_partial_shape();
        int dyn_axis = -1;
        if (ps.rank().is_static()) {
            std::vector<int> candidates;
            for (size_t d = 0; d < ps.size(); ++d) {
                const bool axis_growable = ps[d].is_dynamic() || (ps[d].is_static() && ps[d].get_length() == 0);
                if (axis_growable) {
                    candidates.push_back(static_cast<int>(d));
                }
            }
            if (candidates.size() == 1) {
                dyn_axis = candidates[0];
            } else if (candidates.size() > 1) {
                // Prefer unique non-batch axis (KV-cache layout: [batch, heads, seq, dim]).
                int non_batch = 0;
                int picked = -1;
                for (int c : candidates) {
                    if (c != 0) {
                        ++non_batch;
                        picked = c;
                    }
                }
                if (non_batch == 1) {
                    dyn_axis = picked;
                } else {
                    axis_ambiguous = true;  // multiple non-batch candidates
                }
            }
        }
        if (dyn_axis >= 0) {
            auto shape_of = std::make_shared<v3::ShapeOf>(s0, ov::element::i64);
            auto idx = v0::Constant::create(ov::element::i64, ov::Shape{1}, {dyn_axis});
            auto gather_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
            auto gather = std::make_shared<ov::op::v8::Gather>(shape_of, idx, gather_axis);
            auto squeeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            length_value = std::make_shared<ov::op::v0::Squeeze>(gather, squeeze_axis)->output(0);
        }
    }
    if (!length_value.get_node_shared_ptr()) {
        if (axis_ambiguous) {
            return false;  // ambiguous accumulation axis — leave unresolved
        }
        // Fully-static slot chain: length == slot count.
        const auto n = static_cast<int64_t>(slots->size());
        length_value = v0::Constant::create(ov::element::i64, ov::Shape{}, {n})->output(0);
    }
    len->output(0).replace(length_value);
    return true;
}

void finalize_lowering(const std::shared_ptr<ov::Model>& model) {
    disconnect_dead_sequence_back_edges(model);
    disconnect_dead_sequence_helpers(model);
    align_if_branch_result_ranks(model);
    model->validate_nodes_and_infer_types();
}

}  // namespace

}  // namespace sal_detail

bool SequenceArrayLowering::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (!model) {
        return false;
    }
    // Skip: ConcatFromSequence patterns don't use slot-based lowering.
    if (!sal_detail::model_has_sequence_reader(model)) {
        return false;
    }
    sal_detail::SlotResolver resolver(model);

    bool overall_changed = false;
    bool iter_changed = true;
    while (iter_changed) {
        std::vector<std::shared_ptr<ov::Node>> helpers;
        sal_detail::collect_helpers_recursive(model, helpers);

        iter_changed = false;
        for (const auto& h : helpers) {
            if (h->output(0).get_target_inputs().empty()) {
                continue;
            }
            if (auto at = ov::as_type_ptr<ov::frontend::SequenceAt>(h)) {
                iter_changed |= sal_detail::lower_sequence_at(at, resolver);
            } else if (auto len = ov::as_type_ptr<ov::frontend::SequenceLength>(h)) {
                iter_changed |= sal_detail::lower_sequence_length(len, resolver);
            }
        }
        overall_changed |= iter_changed;
    }

    resolver.finalize_pending_wiring();

    if (overall_changed || resolver.changed()) {
        sal_detail::finalize_lowering(model);
    }
    return overall_changed || resolver.changed();
}

}  // namespace pass
}  // namespace frontend
}  // namespace ov
