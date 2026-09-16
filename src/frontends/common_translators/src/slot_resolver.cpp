// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "slot_resolver.hpp"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <vector>

#include "openvino/core/except.hpp"
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
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/util/log.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace pass {
namespace sal_detail {

// Shape-preserving select using v8::If; unlike v1::Select, does not broadcast inputs.
ov::Output<ov::Node> make_shape_preserving_select(const ov::Output<ov::Node>& cond_in,
                                                  const ov::Output<ov::Node>& then_val,
                                                  const ov::Output<ov::Node>& else_val) {
    ov::Output<ov::Node> cond = cond_in;
    if (cond.get_element_type() != ov::element::boolean) {
        cond = std::make_shared<v0::Convert>(cond, ov::element::boolean)->output(0);
    }
    if (cond.get_partial_shape().rank().is_static() && cond.get_partial_shape().rank().get_length() > 0) {
        std::vector<int64_t> axes(cond.get_partial_shape().rank().get_length());
        std::iota(axes.begin(), axes.end(), 0);
        auto axes_c = v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
        cond = std::make_shared<v0::Squeeze>(cond, axes_c)->output(0);
    }
    auto then_param = std::make_shared<v0::Parameter>(then_val.get_element_type(), then_val.get_partial_shape());
    auto else_param = std::make_shared<v0::Parameter>(else_val.get_element_type(), else_val.get_partial_shape());
    auto then_result = std::make_shared<v0::Result>(then_param);
    auto else_result = std::make_shared<v0::Result>(else_param);
    auto then_body = std::make_shared<ov::Model>(ov::ResultVector{then_result}, ov::ParameterVector{then_param});
    auto else_body = std::make_shared<ov::Model>(ov::ResultVector{else_result}, ov::ParameterVector{else_param});
    auto if_node = std::make_shared<v8::If>(cond);
    if_node->set_then_body(then_body);
    if_node->set_else_body(else_body);
    if_node->set_input(then_val, then_param, nullptr);
    if_node->set_input(else_val, nullptr, else_param);
    if_node->set_output(then_result, else_result);
    return if_node->output(0);
}

ov::Output<ov::Node> unwrap_identity(const ov::Output<ov::Node>& value) {
    ov::Output<ov::Node> cur = value;
    while (auto identity = ov::as_type_ptr<v16::Identity>(cur.get_node_shared_ptr())) {
        cur = identity->input_value(0);
    }
    return cur;
}

ov::Output<ov::Node> make_zero_dummy(const ov::Output<ov::Node>& tmpl) {
    auto et = tmpl.get_element_type();
    auto ps = tmpl.get_partial_shape();
    if (et == ov::element::dynamic) {
        et = ov::element::i64;
    }
    std::vector<int64_t> dims;
    if (!ps.rank().is_dynamic()) {
        for (const auto& d : ps) {
            dims.push_back(d.is_dynamic() ? 0 : d.get_length());
        }
    }
    ov::Shape shape(dims.begin(), dims.end());
    return std::make_shared<v0::Constant>(et, shape, std::vector<std::string>{"0"});
}

// Seed constant with static dims preserved, last dynamic axis set to 0
// (growable concat axis), all other dynamic axes set to 1.
ov::Output<ov::Node> make_growable_seed(const ov::PartialShape& ps, ov::element::Type et) {
    if (et == ov::element::dynamic) {
        et = ov::element::i64;
    }
    if (!ps.rank().is_static()) {
        return std::make_shared<v0::Constant>(et, ov::Shape{}, std::vector<std::string>{"0"})->output(0);
    }
    const auto rl = ps.rank().get_length();
    int64_t last_dyn = -1;
    for (int64_t j = rl - 1; j >= 0; --j) {
        if (ps[j].is_dynamic()) {
            last_dyn = j;
            break;
        }
    }
    ov::Shape seed_shape;
    seed_shape.reserve(static_cast<size_t>(rl));
    for (int64_t j = 0; j < rl; ++j) {
        if (ps[j].is_static()) {
            seed_shape.push_back(static_cast<size_t>(ps[j].get_length()));
        } else if (j == last_dyn) {
            seed_shape.push_back(0);
        } else {
            seed_shape.push_back(1);
        }
    }
    // Single-value literal broadcasts to fill the shape.
    return std::make_shared<v0::Constant>(et, seed_shape, std::vector<std::string>{"0"})->output(0);
}

namespace {

bool depends_on_node(const ov::Output<ov::Node>& value, ov::Node* target, std::set<ov::Node*>& visited) {
    auto node = value.get_node();
    if (node == target) {
        return true;
    }
    if (!visited.insert(node).second) {
        return false;
    }
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        if (depends_on_node(node->input_value(i), target, visited)) {
            return true;
        }
    }
    return false;
}

// Returns index of value's Result in body, adding one if absent.
int64_t ensure_body_result(const std::shared_ptr<ov::Model>& body, const ov::Output<ov::Node>& value) {
    int64_t idx = body->get_result_index(value);
    if (idx < 0) {
        auto r = std::make_shared<v0::Result>(value);
        body->add_results({r});
        idx = static_cast<int64_t>(body->get_results().size()) - 1;
    }
    return idx;
}

struct InputBinding {
    int outer_input = -1;
    int back_edge_result_idx = -1;  // >= 0 only for MergedInputDescription
};

bool find_input_binding(const std::shared_ptr<ov::op::util::MultiSubGraphOp>& msg,
                        int body_idx,
                        size_t param_idx,
                        InputBinding& out) {
    for (const auto& d : msg->get_input_descriptions(body_idx)) {
        if (d->m_body_parameter_index != param_idx) {
            continue;
        }
        out.outer_input = static_cast<int>(d->m_input_index);
        if (auto merged = ov::as_type_ptr<ov::op::util::MultiSubGraphOp::MergedInputDescription>(d)) {
            out.back_edge_result_idx = static_cast<int>(merged->m_body_value_index);
        }
        return true;
    }
    return false;
}

}  // namespace

void SlotResolver::build_maps(const std::shared_ptr<ov::Model>& m) {
    if (!m) {
        return;
    }
    for (const auto& p : m->get_parameters()) {
        param_to_model_[p.get()] = m;
    }
    for (const auto& node : m->get_ordered_ops()) {
        if (auto msg = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
            for (size_t i = 0; i < msg->get_internal_subgraphs_size(); ++i) {
                auto body = msg->get_function(i);
                if (body) {
                    body_owner_[body.get()] = {msg, static_cast<int>(i)};
                    build_maps(body);
                }
            }
        }
    }
}

// True when the Loop merged input carries a sequence (outer source or back-edge is
// a sequence helper, or the body Parameter feeds one).
bool SlotResolver::is_sequence_merged_input(const std::shared_ptr<v5::Loop>& loop,
                                            const std::shared_ptr<ov::Model>& body,
                                            size_t p_idx,
                                            size_t back_value_idx) {
    InputBinding binding;
    const int outer_input = find_input_binding(loop, 0, p_idx, binding) ? binding.outer_input : -1;
    if (outer_input >= 0) {
        auto src = unwrap_identity(loop->input_value(outer_input));
        auto node = src.get_node_shared_ptr();
        if (ov::is_type<ov::frontend::SequenceMark>(node) || ov::is_type<ov::frontend::SequenceInsert>(node) ||
            ov::is_type<ov::frontend::SequenceErase>(node)) {
            return true;
        }
    }
    if (back_value_idx < body->get_results().size()) {
        auto back_src = unwrap_identity(body->get_results()[back_value_idx]->input_value(0));
        auto back_node = back_src.get_node_shared_ptr();
        if (ov::is_type<ov::frontend::SequenceMark>(back_node) ||
            ov::is_type<ov::frontend::SequenceInsert>(back_node) ||
            ov::is_type<ov::frontend::SequenceErase>(back_node)) {
            return true;
        }
    }
    const auto& body_param = body->get_parameters()[p_idx];
    std::deque<ov::Output<ov::Node>> q;
    std::set<ov::Node*> visited;
    q.push_back(body_param->output(0));
    while (!q.empty()) {
        auto v = q.front();
        q.pop_front();
        for (const auto& tin : v.get_target_inputs()) {
            auto consumer = tin.get_node();
            if (!visited.insert(consumer).second) {
                continue;
            }
            if (ov::is_type<v16::Identity>(consumer)) {
                q.push_back(consumer->output(0));
                continue;
            }
            if (ov::is_type<ov::frontend::SequenceAt>(consumer) ||
                ov::is_type<ov::frontend::SequenceInsert>(consumer) ||
                ov::is_type<ov::frontend::SequenceErase>(consumer) ||
                ov::is_type<ov::frontend::SequenceLength>(consumer)) {
                return true;
            }
        }
    }
    return false;
}

// Pre-allocate N per-slot merged inputs for every Loop whose merged input carries
// a sequence. Defers actual wiring to finalize_pending_wiring().
//
//   merged( seqParam <- seqResult )  ==>  merged( s0Param <- s0Result )
//                                         merged( s1Param <- s1Result ) ...
void SlotResolver::preallocate_loop_merged_params(const std::shared_ptr<ov::Model>& m) {
    if (!m) {
        return;
    }
    for (const auto& node : m->get_ordered_ops()) {
        if (auto msg = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
            for (size_t i = 0; i < msg->get_internal_subgraphs_size(); ++i) {
                preallocate_loop_merged_params(msg->get_function(i));
            }
        }
    }
    for (const auto& node : m->get_ordered_ops()) {
        auto loop = ov::as_type_ptr<v5::Loop>(node);
        if (!loop) {
            continue;
        }
        auto body = loop->get_function();
        if (!body) {
            continue;
        }
        // Snapshot descriptors so set_merged_input mutations don't race with later iterations.
        auto descriptors = loop->get_input_descriptions();
        for (const auto& d : descriptors) {
            auto merged = ov::as_type_ptr<ov::op::util::MultiSubGraphOp::MergedInputDescription>(d);
            if (!merged) {
                continue;
            }
            const size_t p_idx = merged->m_body_parameter_index;
            const size_t back_idx = merged->m_body_value_index;
            if (p_idx >= body->get_parameters().size() || back_idx >= body->get_results().size()) {
                continue;
            }
            if (!is_sequence_merged_input(loop, body, p_idx, back_idx)) {
                continue;
            }
            auto old_param = body->get_parameters()[p_idx];
            if (cache_.count(old_param->output(0))) {
                continue;  // already pre-allocated
            }

            const int outer_in = static_cast<int>(merged->m_input_index);
            auto outer_src = unwrap_identity(loop->input_value(outer_in));
            auto outer_mark = ov::as_type_ptr<ov::frontend::SequenceMark>(outer_src.get_node_shared_ptr());

            std::vector<ov::Output<ov::Node>> slot_templates;
            Slots outer_seed_slots;
            PendingMerged::SeedKind seed_kind = PendingMerged::SeedKind::DEFERRED;

            if (outer_mark && outer_mark->get_input_size() > 0) {
                for (size_t k = 0; k < outer_mark->get_input_size(); ++k) {
                    slot_templates.push_back(outer_mark->input_value(k));
                    outer_seed_slots.push_back(outer_mark->input_value(k));
                }
                seed_kind = PendingMerged::SeedKind::KNOWN;
            } else if (auto os = slots_of(outer_src); os && !os->empty()) {
                // Outer source has priority for slot count (inner back-edge may undercount).
                slot_templates.assign(os->begin(), os->end());
                outer_seed_slots = *os;
                seed_kind = PendingMerged::SeedKind::KNOWN;
            } else {
                auto back_value = body->get_results()[back_idx]->input_value(0);

                if (auto bs = slots_of(back_value)) {
                    slot_templates = *bs;
                } else if (auto tmpl = find_template_via_chain(back_value, old_param.get())) {
                    slot_templates = tmpl->slot_templates;
                } else {
                    continue;  // cannot infer N -- skip
                }
                if (outer_mark && outer_mark->get_input_size() == 0) {
                    for (const auto& t : slot_templates) {
                        outer_seed_slots.push_back(make_growable_seed(t.get_partial_shape(), t.get_element_type()));
                    }
                    seed_kind = PendingMerged::SeedKind::SYNTHETIC;
                }
            }

            const size_t N = slot_templates.size();

            // Fallback element type: first resolved type across all slots.
            ov::element::Type merged_et = ov::element::dynamic;
            for (const auto& t : slot_templates) {
                if (merged_et == ov::element::dynamic) {
                    merged_et = t.get_element_type();
                }
            }

            std::vector<std::shared_ptr<v0::Parameter>> new_params;
            new_params.reserve(N);
            Slots param_outputs;
            param_outputs.reserve(N);
            for (size_t k = 0; k < N; ++k) {
                auto et = slot_templates[k].get_element_type();
                if (et == ov::element::dynamic) {
                    et = merged_et;
                }
                // Widen zero dims to dynamic to avoid clamping the back-edge at runtime.
                ov::PartialShape pshape = slot_templates[k].get_partial_shape();
                if (pshape.rank().is_static()) {
                    for (size_t d = 0; d < pshape.size(); ++d) {
                        if (pshape[d].is_static() && pshape[d].get_length() == 0) {
                            pshape[d] = ov::Dimension::dynamic();
                        }
                    }
                }
                auto np = std::make_shared<v0::Parameter>(et, pshape);
                body->add_parameters({np});
                param_to_model_[np.get()] = body;
                if (seed_kind == PendingMerged::SeedKind::SYNTHETIC) {
                    // Mark as empty-seed so SequenceLength lowers to runtime ShapeOf.
                    np->get_rt_info()["sal_empty_seed_slot"] = true;
                }
                new_params.push_back(np);
                param_outputs.push_back(np->output(0));
            }
            cache_[old_param->output(0)] = param_outputs;

            PendingMerged pm;
            pm.loop = loop;
            pm.body = body;
            pm.back_edge_result_idx = static_cast<int>(back_idx);
            pm.outer_input = outer_in;
            pm.old_param = old_param;
            pm.new_params = std::move(new_params);
            pm.outer_seed_slots = std::move(outer_seed_slots);
            pm.seed_kind = seed_kind;
            pending_merged_.push_back(std::move(pm));
        }
    }
}

void SlotResolver::finalize_pending_wiring() {
    for (auto& pm : pending_merged_) {
        const size_t N = pm.new_params.size();
        if (N == 0) {
            continue;
        }
        // Resolve outer seed slots deferred from pre-allocation time.
        if (pm.seed_kind == PendingMerged::SeedKind::DEFERRED) {
            auto outer_src = pm.loop->input_value(pm.outer_input);
            auto src_slots = slots_of(outer_src);
            OPENVINO_ASSERT(src_slots && src_slots->size() == N,
                            "SequenceArrayLowering: could not reconcile the loop-carried sequence outer seed to ",
                            N,
                            " slots; the loop-carried sequence pattern is not supported.");
            pm.outer_seed_slots = *src_slots;
            pm.seed_kind = PendingMerged::SeedKind::KNOWN;
        }
        auto back_value = pm.body->get_results()[pm.back_edge_result_idx]->input_value(0);
        auto back_slots = slots_of(back_value);
        OPENVINO_ASSERT(back_slots && back_slots->size() == N,
                        "SequenceArrayLowering: could not reconcile the loop-carried sequence back-edge to ",
                        N,
                        " slots; the loop-carried sequence pattern is not supported.");
        // Synthetic seed was built from pre-lowering templates (may be scalar);
        // rebuild from resolved back-edge shape so the carried tensor keeps its true rank.
        if (pm.seed_kind == PendingMerged::SeedKind::SYNTHETIC) {
            for (size_t k = 0; k < N; ++k) {
                const auto& resolved = (*back_slots)[k];
                if (resolved.get_partial_shape().rank().is_dynamic()) {
                    if (auto producer = resolved.get_node_shared_ptr()) {
                        producer->validate_and_infer_types();
                    }
                }
                pm.outer_seed_slots[k] = make_growable_seed(resolved.get_partial_shape(), resolved.get_element_type());
            }
        }
        for (size_t k = 0; k < N; ++k) {
            ov::Output<ov::Node> back_value = (*back_slots)[k];
            ensure_body_result(pm.body, back_value);
            pm.loop->set_merged_input(pm.new_params[k], pm.outer_seed_slots[k], back_value);
        }
        pm.loop->input(pm.outer_input).replace_source_output(pm.outer_seed_slots[0]);
        changed_ = true;
    }
}

std::optional<LengthTemplate> SlotResolver::find_template_via_chain(const ov::Output<ov::Node>& root_value,
                                                                    ov::Node* exclude_p) {
    std::deque<ov::Output<ov::Node>> q;
    std::set<ov::Output<ov::Node>> visited;
    q.push_back(root_value);
    while (!q.empty()) {
        auto v = q.front();
        q.pop_front();
        if (!visited.insert(v).second) {
            continue;
        }
        auto n = v.get_node_shared_ptr();
        if (auto mark = ov::as_type_ptr<ov::frontend::SequenceMark>(n)) {
            if (n.get() != exclude_p && mark->get_input_size() > 0) {
                bool deps = false;
                for (size_t i = 0; i < mark->get_input_size(); ++i) {
                    std::set<ov::Node*> vis;
                    if (depends_on_node(mark->input_value(i), exclude_p, vis)) {
                        deps = true;
                        break;
                    }
                }
                if (!deps) {
                    // Enumerate structurally (not via slots_of()) to avoid cyclic dependency.
                    LengthTemplate t;
                    t.slot_templates = mark->get_sequence();
                    return t;
                }
            }
            continue;
        }
        if (auto msg = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(n)) {
            for (size_t b = 0; b < msg->get_internal_subgraphs_size(); ++b) {
                for (const auto& d : msg->get_output_descriptions(static_cast<int>(b))) {
                    if (d->m_output_index == v.get_index()) {
                        auto body = msg->get_function(b);
                        if (body) {
                            q.push_back(body->get_results()[d->m_body_value_index]->input_value(0));
                        }
                    }
                }
            }
            continue;
        }
        if (auto param = ov::as_type_ptr<v0::Parameter>(n)) {
            auto it = param_to_model_.find(param.get());
            if (it == param_to_model_.end()) {
                continue;
            }
            auto body = it->second;
            auto own_it = body_owner_.find(body.get());
            if (own_it == body_owner_.end()) {
                continue;
            }
            auto owner_msg = own_it->second.first;
            int body_idx = own_it->second.second;
            const int64_t pi_signed = it->second->get_parameter_index(param);
            if (pi_signed < 0) {
                continue;
            }
            InputBinding binding;
            if (find_input_binding(owner_msg, body_idx, static_cast<size_t>(pi_signed), binding)) {
                q.push_back(owner_msg->input_value(binding.outer_input));
                if (n.get() != exclude_p && binding.back_edge_result_idx >= 0) {
                    q.push_back(it->second->get_results()[binding.back_edge_result_idx]->input_value(0));
                }
            }
            continue;
        }
        if (ov::as_type_ptr<ov::frontend::SequenceInsert>(n) || ov::as_type_ptr<ov::frontend::SequenceErase>(n)) {
            q.push_back(n->input_value(0));
            continue;
        }
        for (size_t i = 0; i < n->get_input_size(); ++i) {
            q.push_back(n->input_value(i));
        }
    }
    return std::nullopt;
}

std::optional<Slots> SlotResolver::slots_of(const ov::Output<ov::Node>& value_in) {
    auto value = unwrap_identity(value_in);
    auto it = cache_.find(value);
    if (it != cache_.end()) {
        return it->second;
    }
    if (!in_progress_.insert(value).second) {
        return std::nullopt;  // cycle - cannot resolve via this path
    }
    struct Guard {
        std::set<ov::Output<ov::Node>>* set;
        ov::Output<ov::Node> v;
        ~Guard() {
            set->erase(v);
        }
    } guard{&in_progress_, value};

    auto node = value.get_node_shared_ptr();
    std::optional<Slots> result;

    if (auto mark = ov::as_type_ptr<ov::frontend::SequenceMark>(node)) {
        Slots s;
        s.reserve(mark->get_input_size());
        for (size_t i = 0; i < mark->get_input_size(); ++i) {
            auto in = mark->input_value(i);
            auto in_node = in.get_node_shared_ptr();
            const bool seq_typed = ov::as_type_ptr<ov::frontend::SequenceInsert>(in_node) ||
                                   ov::as_type_ptr<ov::frontend::SequenceErase>(in_node) ||
                                   ov::as_type_ptr<ov::frontend::SequenceMark>(in_node);
            if (seq_typed) {
                if (auto inner = slots_of(in)) {
                    s.insert(s.end(), inner->begin(), inner->end());
                    continue;
                }
            }
            s.push_back(in);
        }
        result = std::move(s);
    } else if (auto ins = ov::as_type_ptr<ov::frontend::SequenceInsert>(node)) {
        auto base = slots_of(ins->input_value(0));
        if (!base) {
            return std::nullopt;
        }
        Slots s = *base;
        if (!ins->has_position()) {
            s.push_back(ins->input_value(1));
            result = std::move(s);
        } else {
            auto pos_c = ov::util::get_constant_from_source(ins->input_value(2));
            if (!pos_c) {
                // out_j = (j < pos) ? base[j] : (j == pos) ? value : base[j-1]
                const auto& val = ins->input_value(1);
                const auto& pos_in = ins->input_value(2);
                auto pos_i64 = std::make_shared<v0::Convert>(pos_in, element::i64);
                Slots s_new;
                s_new.reserve(base->size() + 1);
                for (size_t j = 0; j <= base->size(); ++j) {
                    auto jc = v0::Constant::create(element::i64, ov::Shape{}, {static_cast<int64_t>(j)});
                    auto less = std::make_shared<v1::Less>(jc, pos_i64);
                    auto eq = std::make_shared<v1::Equal>(jc, pos_i64);
                    ov::Output<ov::Node> from_lo =
                        base->empty() ? val : (j < base->size() ? base->at(j) : base->at(base->size() - 1));
                    ov::Output<ov::Node> from_hi = base->empty() ? val : ((j > 0) ? base->at(j - 1) : base->at(0));
                    auto inner = make_shape_preserving_select(eq, val, from_hi);
                    auto outer = make_shape_preserving_select(less, from_lo, inner);
                    s_new.push_back(outer);
                }
                result = std::move(s_new);
            } else {
                const auto pv = pos_c->cast_vector<int64_t>();
                if (pv.size() != 1) {
                    return std::nullopt;
                }
                int64_t pos = pv[0];
                if (pos < 0) {
                    // Negative index wraps: pos += size.
                    pos += static_cast<int64_t>(base->size());
                }
                if (pos < 0 || static_cast<size_t>(pos) > base->size()) {
                    return std::nullopt;
                }
                s.insert(s.begin() + pos, ins->input_value(1));
                result = std::move(s);
            }
        }
    } else if (auto era = ov::as_type_ptr<ov::frontend::SequenceErase>(node)) {
        auto base = slots_of(era->input_value(0));
        if (!base) {
            return std::nullopt;
        }
        Slots s = *base;
        if (s.empty()) {
            return std::nullopt;
        }
        if (!era->has_position()) {
            s.pop_back();
            result = std::move(s);
        } else {
            auto pos_c = ov::util::get_constant_from_source(era->input_value(1));
            if (!pos_c) {
                // Dynamic position: synthesize each output slot via Select.
                // out_j = (j < pos) ? base[j] : base[j + 1]
                const auto& pos_in = era->input_value(1);
                auto pos_i64 = std::make_shared<v0::Convert>(pos_in, element::i64);
                Slots s_new;
                s_new.reserve(base->size() - 1);
                for (size_t j = 0; j + 1 < base->size(); ++j) {
                    auto jc = v0::Constant::create(element::i64, ov::Shape{}, {static_cast<int64_t>(j)});
                    auto less = std::make_shared<v1::Less>(jc, pos_i64);
                    auto sel = make_shape_preserving_select(less, base->at(j), base->at(j + 1));
                    s_new.push_back(sel);
                }
                result = std::move(s_new);
            } else {
                const auto pv = pos_c->cast_vector<int64_t>();
                if (pv.size() != 1) {
                    return std::nullopt;
                }
                int64_t pos = pv[0];
                if (pos < 0) {
                    pos += static_cast<int64_t>(base->size());
                }
                if (pos < 0 || static_cast<size_t>(pos) >= base->size()) {
                    return std::nullopt;
                }
                s.erase(s.begin() + pos);
                result = std::move(s);
            }
        }
    } else if (auto p = ov::as_type_ptr<v0::Parameter>(node)) {
        result = slots_of_param(p);
    } else if (auto msg = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
        result = slots_of_msg_output(msg, value.get_index());
    }

    if (result) {
        cache_[value] = *result;
    }
    return result;
}

bool SlotResolver::is_loop_carried_empty_seed(const ov::Output<ov::Node>& value) {
    auto s = slots_of(value);
    if (!s) {
        return false;
    }
    for (const auto& slot : *s) {
        if (unwrap_identity(slot).get_node()->get_rt_info().count("sal_empty_seed_slot")) {
            return true;
        }
    }
    return false;
}

std::optional<Slots> SlotResolver::slots_of_param(const std::shared_ptr<v0::Parameter>& p) {
    auto body_it = param_to_model_.find(p.get());
    if (body_it == param_to_model_.end()) {
        return std::nullopt;  // not in any tracked body - top-level Parameter
    }
    auto body = body_it->second;
    auto owner_it = body_owner_.find(body.get());
    if (owner_it == body_owner_.end()) {
        return std::nullopt;
    }
    auto owner = owner_it->second.first;
    const int body_idx = owner_it->second.second;

    const int64_t p_idx_signed = body->get_parameter_index(p);
    if (p_idx_signed < 0) {
        return std::nullopt;
    }
    const size_t p_idx = static_cast<size_t>(p_idx_signed);

    InputBinding binding;
    if (!find_input_binding(owner, body_idx, p_idx, binding)) {
        return std::nullopt;
    }
    const int outer_input = binding.outer_input;
    const int back_edge_result_idx = binding.back_edge_result_idx;

    auto outer_source = owner->input_value(outer_input);
    auto loop = ov::as_type_ptr<v5::Loop>(owner);

    if (loop && back_edge_result_idx >= 0) {
        // Merged inputs are pre-allocated; bail out to avoid cyclic back-edge resolution.
        return std::nullopt;
    }

    auto src_slots = slots_of(outer_source);
    if (!src_slots) {
        return std::nullopt;
    }
    const size_t N = src_slots->size();
    Slots param_outputs;
    param_outputs.reserve(N);

    if (auto subgraph_op = ov::as_type_ptr<ov::op::util::SubGraphOp>(owner)) {
        for (size_t k = 0; k < N; ++k) {
            auto new_param = std::make_shared<v0::Parameter>((*src_slots)[k].get_element_type(),
                                                             (*src_slots)[k].get_partial_shape());
            body->add_parameters({new_param});
            param_to_model_[new_param.get()] = body;
            if ((*src_slots)[k].get_node()->get_rt_info().count("sal_empty_seed_slot")) {
                new_param->get_rt_info()["sal_empty_seed_slot"] = true;
            }
            subgraph_op->set_invariant_input(new_param, (*src_slots)[k]);
            param_outputs.push_back(new_param->output(0));
        }
        owner->input(outer_input).replace_source_output((*src_slots)[0]);
        changed_ = true;
        return param_outputs;
    }

    const size_t num_bodies = owner->get_internal_subgraphs_size();
    std::vector<std::shared_ptr<ov::Model>> bodies(num_bodies);
    std::vector<int> other_param_idx(num_bodies, -1);
    for (size_t bb = 0; bb < num_bodies; ++bb) {
        bodies[bb] = owner->get_function(bb);
        if (!bodies[bb]) {
            return std::nullopt;
        }
        if (static_cast<int>(bb) == body_idx) {
            other_param_idx[bb] = static_cast<int>(p_idx);
            continue;
        }
        for (const auto& d : owner->get_input_descriptions(static_cast<int>(bb))) {
            if (static_cast<int>(d->m_input_index) == outer_input) {
                other_param_idx[bb] = static_cast<int>(d->m_body_parameter_index);
                break;
            }
        }
    }

    std::vector<Slots> per_body_outputs(num_bodies);
    for (size_t k = 0; k < N; ++k) {
        ov::ParameterVector pv;
        for (size_t bb = 0; bb < num_bodies; ++bb) {
            if (other_param_idx[bb] < 0) {
                continue;
            }
            auto new_param = std::make_shared<v0::Parameter>((*src_slots)[k].get_element_type(),
                                                             (*src_slots)[k].get_partial_shape());
            bodies[bb]->add_parameters({new_param});
            param_to_model_[new_param.get()] = bodies[bb];
            if ((*src_slots)[k].get_node()->get_rt_info().count("sal_empty_seed_slot")) {
                new_param->get_rt_info()["sal_empty_seed_slot"] = true;
            }
            pv.push_back(new_param);
            per_body_outputs[bb].push_back(new_param->output(0));
        }
        owner->set_invariant_inputs((*src_slots)[k], pv);
    }
    for (size_t bb = 0; bb < num_bodies; ++bb) {
        if (static_cast<int>(bb) == body_idx || other_param_idx[bb] < 0) {
            continue;
        }
        const auto& other_param = bodies[bb]->get_parameters()[other_param_idx[bb]];
        cache_[other_param->output(0)] = per_body_outputs[bb];
    }
    owner->input(outer_input).replace_source_output((*src_slots)[0]);
    changed_ = true;
    return per_body_outputs[body_idx];
}

bool SlotResolver::expand_branch_to_n_slots(const std::shared_ptr<ov::op::util::MultiSubGraphOp>& msg,
                                            size_t b,
                                            size_t ref,
                                            size_t N,
                                            std::vector<Slots>& per_body_slots) {
    const size_t orig_branch_size = per_body_slots[b].size();
    auto ref_body = msg->get_function(static_cast<int>(ref));
    auto cur_body = msg->get_function(static_cast<int>(b));
    per_body_slots[b].clear();
    per_body_slots[b].reserve(N);
    std::vector<size_t> live_param_indices;
    if (ref_body && cur_body && !per_body_slots[ref].empty()) {
        const auto ref_et = per_body_slots[ref][0].get_element_type();
        const auto ref_ps = per_body_slots[ref][0].get_partial_shape();
        const auto& ref_params = ref_body->get_parameters();
        for (size_t i = 0; i < ref_params.size(); ++i) {
            const auto& p = ref_params[i];
            if (p->get_element_type() != ref_et) {
                continue;
            }
            if (p->get_partial_shape().rank() != ref_ps.rank()) {
                continue;
            }
            bool compatible = true;
            if (!ref_ps.rank().is_dynamic()) {
                const auto& pps = p->get_partial_shape();
                for (size_t d = 0; d < ref_ps.size(); ++d) {
                    const auto& rd = ref_ps[d];
                    const auto& pd = pps[d];
                    if (rd.is_static() && pd.is_static() && rd.get_length() != pd.get_length()) {
                        compatible = false;
                        break;
                    }
                }
            }
            if (compatible) {
                live_param_indices.push_back(i);
            }
        }
    }
    // Prefer the last N (SAL-appended Parameters are at the tail).
    bool use_mirror = false;
    std::vector<size_t> mirror_live_param_indices;
    std::vector<int64_t> mirror_outer_input_index;
    if (live_param_indices.size() >= N) {
        mirror_live_param_indices.assign(live_param_indices.end() - static_cast<long>(N), live_param_indices.end());
        mirror_outer_input_index.assign(N, -1);
        auto live_descs = msg->get_input_descriptions(static_cast<int>(ref));
        for (size_t k = 0; k < N; ++k) {
            for (const auto& d : live_descs) {
                if (d->m_body_parameter_index == mirror_live_param_indices[k]) {
                    mirror_outer_input_index[k] = static_cast<int64_t>(d->m_input_index);
                    break;
                }
            }
            if (mirror_outer_input_index[k] < 0) {
                mirror_outer_input_index.clear();
                break;
            }
        }
        use_mirror = !mirror_outer_input_index.empty();
    }
    if (use_mirror) {
        auto cur_descs = msg->get_input_descriptions(static_cast<int>(b));
        for (size_t k = 0; k < N; ++k) {
            const auto& live_param = ref_body->get_parameters()[mirror_live_param_indices[k]];
            auto new_param =
                std::make_shared<v0::Parameter>(live_param->get_element_type(), live_param->get_partial_shape());
            cur_body->add_parameters({new_param});
            param_to_model_[new_param.get()] = cur_body;
            if (live_param->get_rt_info().count("sal_empty_seed_slot")) {
                new_param->get_rt_info()["sal_empty_seed_slot"] = true;
            }
            const size_t new_body_param_idx = cur_body->get_parameters().size() - 1;
            cur_descs.push_back(std::make_shared<ov::op::util::MultiSubGraphOp::InvariantInputDescription>(
                static_cast<uint64_t>(mirror_outer_input_index[k]),
                static_cast<uint64_t>(new_body_param_idx)));
            per_body_slots[b].push_back(new_param->output(0));
        }
        msg->set_input_descriptions(static_cast<int>(b), cur_descs);
        return true;
    }
    // Opaque-forward branch (size 1): filling with zeros would corrupt a live KV cache. Leave unlowered.
    // Empty branch (size 0) is safe to expand with zeros.
    if (orig_branch_size == 1) {
        return false;
    }
    for (size_t k = 0; k < N; ++k) {
        auto dummy = make_zero_dummy(per_body_slots[ref][k]);
        (void)cur_body;
        per_body_slots[b].push_back(dummy);
    }
    return true;
}

std::optional<Slots> SlotResolver::slots_of_msg_output(const std::shared_ptr<ov::op::util::MultiSubGraphOp>& msg,
                                                       size_t out_idx) {
    const size_t num_bodies = msg->get_internal_subgraphs_size();
    std::vector<Slots> per_body_slots;
    per_body_slots.reserve(num_bodies);
    for (size_t b = 0; b < num_bodies; ++b) {
        int result_idx = -1;
        for (const auto& d : msg->get_output_descriptions(static_cast<int>(b))) {
            if (d->m_output_index == out_idx) {
                result_idx = static_cast<int>(d->m_body_value_index);
                break;
            }
        }
        if (result_idx < 0) {
            return std::nullopt;
        }
        auto body = msg->get_function(b);
        if (!body) {
            return std::nullopt;
        }
        auto body_value = body->get_results()[result_idx]->input_value(0);
        auto s = slots_of(body_value);
        if (!s) {
            return std::nullopt;
        }
        per_body_slots.push_back(std::move(*s));
    }
    if (per_body_slots.empty()) {
        return std::nullopt;
    }
    // Reconcile slot counts across branches.
    size_t N = 0;
    for (const auto& s : per_body_slots) {
        N = std::max(N, s.size());
    }
    if (N == 0) {
        return std::nullopt;
    }
    for (size_t b = 0; b < per_body_slots.size(); ++b) {
        if (per_body_slots[b].size() == N) {
            continue;
        }
        // size 1: opaque-forward; size > 1 and < N: genuine cross-branch length mismatch.
        if (per_body_slots[b].size() > 1 && per_body_slots[b].size() != N) {
            return std::nullopt;
        }
        size_t ref = 0;
        for (size_t bb = 0; bb < per_body_slots.size(); ++bb) {
            if (per_body_slots[bb].size() == N) {
                ref = bb;
                break;
            }
        }
        if (!expand_branch_to_n_slots(msg, b, ref, N, per_body_slots)) {
            return std::nullopt;
        }
    }
    for (const auto& s : per_body_slots) {
        if (s.size() != N) {
            return std::nullopt;
        }
    }

    Slots outer_outputs;
    outer_outputs.reserve(N);
    if (auto if_op = ov::as_type_ptr<v8::If>(msg)) {
        for (size_t k = 0; k < N; ++k) {
            auto then_r = std::make_shared<v0::Result>(per_body_slots[0][k]);
            auto else_r = std::make_shared<v0::Result>(per_body_slots[1][k]);
            if_op->get_then_body()->add_results({then_r});
            if_op->get_else_body()->add_results({else_r});
            auto new_out = if_op->set_output(then_r, else_r);
            outer_outputs.push_back(new_out);
        }
    } else if (auto loop = ov::as_type_ptr<v5::Loop>(msg)) {
        for (size_t k = 0; k < N; ++k) {
            ensure_body_result(loop->get_function(), per_body_slots[0][k]);
            outer_outputs.push_back(loop->get_iter_value(per_body_slots[0][k]));
        }
    } else {
        return std::nullopt;
    }
    changed_ = true;
    return outer_outputs;
}

}  // namespace sal_detail
}  // namespace pass
}  // namespace frontend
}  // namespace ov
