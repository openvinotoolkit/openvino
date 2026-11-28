// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "recurrent_replay_fusion.hpp"
#include "recurrent_replay_graph.hpp"
#include "intel_gpu/op/recurrent_cache_replay.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace ov::intel_gpu {
namespace {
namespace ops = ov::opset13;
using Subgraph = ov::op::util::SubGraphOp;
using Writebacks = std::unordered_map<const Node*, std::shared_ptr<ops::Assign>>;

constexpr const char* replay_cache_state_prefix = "__ggml_replay_cache_";
const std::vector<size_t>& bank_ports(bool conv) {
    static const std::vector<size_t> convolution{0};
    static const std::vector<size_t> gdn{3, 6, 7, 8};
    return conv ? convolution : gdn;
}

struct Match {
    std::unordered_map<const Node*, Output<Node>> boundary;
    std::unordered_map<const Node*, const Node*> nodes;
};

bool models_match(const std::shared_ptr<Model>& pattern, const std::shared_ptr<Model>& candidate);

bool ports_match(const ops::Loop& lhs, const ops::Loop& rhs) {
    const auto& a = lhs.get_input_descriptions();
    const auto& b = rhs.get_input_descriptions();
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->get_type_info() != b[i]->get_type_info() || a[i]->m_input_index != b[i]->m_input_index ||
            a[i]->m_body_parameter_index != b[i]->m_body_parameter_index)
            return false;
        if (auto merged = ov::as_type_ptr<Subgraph::MergedInputDescription>(a[i])) {
            if (merged->m_body_value_index !=
                ov::as_type_ptr<Subgraph::MergedInputDescription>(b[i])->m_body_value_index)
                return false;
        } else if (!ov::is_type<Subgraph::InvariantInputDescription>(a[i]))
            return false;
    }
    const auto& c = lhs.get_output_descriptions();
    const auto& d = rhs.get_output_descriptions();
    if (c.size() != d.size())
        return false;
    for (size_t i = 0; i < c.size(); ++i) {
        if (c[i]->get_type_info() != d[i]->get_type_info() || c[i]->m_output_index != d[i]->m_output_index ||
            c[i]->m_body_value_index != d[i]->m_body_value_index)
            return false;
        if (auto full = ov::as_type_ptr<Subgraph::BodyOutputDescription>(c[i])) {
            if (full->m_iteration != ov::as_type_ptr<Subgraph::BodyOutputDescription>(d[i])->m_iteration)
                return false;
        } else if (auto x = ov::as_type_ptr<Subgraph::ConcatOutputDescription>(c[i])) {
            auto y = ov::as_type_ptr<Subgraph::ConcatOutputDescription>(d[i]);
            if (x->m_start != y->m_start || x->m_end != y->m_end || x->m_stride != y->m_stride ||
                x->m_part_size != y->m_part_size || x->m_axis != y->m_axis)
                return false;
        } else
            return false;
    }
    return true;
}

bool same_attributes(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
    if (auto x = ov::as_type_ptr<ops::Constant>(a)) {
        auto y = ov::as_type_ptr<ops::Constant>(b);
        return x->get_shape() == y->get_shape() && x->get_byte_size() == y->get_byte_size() &&
               !std::memcmp(x->get_data_ptr(), y->get_data_ptr(), x->get_byte_size());
    }
    if (auto x = ov::as_type_ptr<ops::Gather>(a))
        return x->get_batch_dims() == ov::as_type_ptr<ops::Gather>(b)->get_batch_dims();
    if (auto x = ov::as_type_ptr<ops::Reshape>(a))
        return x->get_special_zero() == ov::as_type_ptr<ops::Reshape>(b)->get_special_zero();
    if (auto x = ov::as_type_ptr<ops::Concat>(a))
        return x->get_axis() == ov::as_type_ptr<ops::Concat>(b)->get_axis();
    if (auto x = ov::as_type_ptr<ops::ReduceSum>(a))
        return x->get_keep_dims() == ov::as_type_ptr<ops::ReduceSum>(b)->get_keep_dims();
    if (auto x = ov::as_type_ptr<ops::Broadcast>(a))
        return x->get_broadcast_spec() == ov::as_type_ptr<ops::Broadcast>(b)->get_broadcast_spec();
    if (auto x = ov::as_type_ptr<ops::GroupConvolution>(a)) {
        auto y = ov::as_type_ptr<ops::GroupConvolution>(b);
        return x->get_strides() == y->get_strides() && x->get_dilations() == y->get_dilations() &&
               x->get_auto_pad() == y->get_auto_pad() && x->get_pads_begin() == y->get_pads_begin() &&
               x->get_pads_end() == y->get_pads_end();
    }
    if (auto x = ov::as_type_ptr<ops::Loop>(a)) {
        auto y = ov::as_type_ptr<ops::Loop>(b);
        const auto px = x->get_special_body_ports(), py = y->get_special_body_ports();
        return px.current_iteration_input_idx == py.current_iteration_input_idx &&
               px.body_condition_output_idx == py.body_condition_output_idx && ports_match(*x, *y) &&
               models_match(x->get_function(), y->get_function());
    }
    // ShapeOf, Range and Convert carry their element type in the output descriptor.
    // The binary ops and Select also carry the broadcast specification.
    return a->get_autob() == b->get_autob() &&
           (ov::is_type<ops::ShapeOf>(a) || ov::is_type<ops::Range>(a) || ov::is_type<ops::Convert>(a) ||
            ov::is_type<ops::Add>(a) || ov::is_type<ops::Multiply>(a) || ov::is_type<ops::Subtract>(a) ||
            ov::is_type<ops::Greater>(a) || ov::is_type<ops::NotEqual>(a) || ov::is_type<ops::Select>(a) ||
            ov::is_type<ops::ScatterNDUpdate>(a) || ov::is_type<ops::Transpose>(a) || ov::is_type<ops::Exp>(a) ||
            ov::is_type<ops::Slice>(a) || ov::is_type<ops::Squeeze>(a) || ov::is_type<ops::Unsqueeze>(a) ||
            ov::is_type<ops::Result>(a));
}

bool outputs_match(const Output<Node>& pattern, const Output<Node>& candidate, Match& match) {
    auto a = pattern.get_node_shared_ptr(), b = candidate.get_node_shared_ptr();
    if (auto boundary = match.boundary.find(a.get()); boundary != match.boundary.end())
        return boundary->second == candidate;
    if (pattern.get_element_type() != candidate.get_element_type())
        return false;
    if (pattern.get_index() != candidate.get_index())
        return false;
    if (auto previous = match.nodes.find(a.get()); previous != match.nodes.end())
        return previous->second == b.get();
    if (a->get_type_info() != b->get_type_info() || a->get_input_size() != b->get_input_size() ||
        !b->get_control_dependencies().empty() || !b->get_control_dependents().empty() || !same_attributes(a, b))
        return false;
    match.nodes.emplace(a.get(), b.get());
    for (size_t i = 0; i < a->get_input_size(); ++i)
        if (!outputs_match(a->input_value(i), b->input_value(i), match))
            return false;
    return true;
}

bool models_match(const std::shared_ptr<Model>& pattern, const std::shared_ptr<Model>& candidate) {
    if (!candidate->get_sinks().empty() || !candidate->get_variables().empty() ||
        pattern->get_parameters().size() != candidate->get_parameters().size() ||
        pattern->get_output_size() != candidate->get_output_size())
        return false;
    Match match;
    for (size_t i = 0; i < pattern->get_parameters().size(); ++i) {
        auto p = pattern->get_parameters()[i], q = candidate->get_parameters()[i];
        if (p->get_element_type() != q->get_element_type() || p->get_partial_shape() != q->get_partial_shape())
            return false;
        match.nodes.emplace(p.get(), q.get());
    }
    for (size_t i = 0; i < pattern->get_output_size(); ++i)
        if (!outputs_match(pattern->get_results()[i], candidate->get_results()[i], match))
            return false;
    return true;
}

const ops::ScatterNDUpdate* bank_write(const Output<Node>& result) {
    auto node = result.get_node();
    for (int axis = 0; axis < 2; ++axis) {
        if (!ov::is_type<ops::Unsqueeze>(node))
            return nullptr;
        node = node->get_input_node_ptr(0);
    }
    return ov::as_type<ops::ScatterNDUpdate>(node);
}

Output<Node> bank_input(const ops::ScatterNDUpdate* write) {
    const auto reset = ov::as_type<ops::ScatterNDUpdate>(write->get_input_node_ptr(0));
    if (!reset)
        return {};
    auto input = reset->input_value(0);
    for (int axis = 0; axis < 2; ++axis) {
        if (!ov::is_type<ops::Squeeze>(input.get_node()))
            return {};
        input = input.get_node()->input_value(0);
    }
    return input;
}

bool rank_four(const Output<Node>& value) {
    return value.get_node() && value.get_partial_shape().rank() == 4;
}

// A bank update always has the same writeback tail. The full match below
// validates the axes, reset, row mapping, and update computation.
Output<Node> bank_update(const ops::ScatterNDUpdate* write) {
    auto values = write->input_value(2);
    if (!ov::is_type<ops::Concat>(values.get_node()))
        return {};
    auto active = values.get_node()->input_value(0);
    return ov::is_type<ops::Reshape>(active.get_node()) ? active.get_node()->input_value(0) : Output<Node>{};
}

const ops::Slice* destination_rows(const ops::ScatterNDUpdate* write) {
    const auto index = ov::as_type<ops::Unsqueeze>(write->get_input_node_ptr(1));
    if (!index || !ov::is_type<ops::Concat>(index->get_input_node_ptr(0)))
        return nullptr;
    return ov::as_type<ops::Slice>(index->get_input_node_ptr(0)->get_input_node_ptr(0));
}

Output<Node> rows_input(const ops::Slice* destination) {
    if (destination) {
        const auto gather = ov::as_type<ops::Gather>(destination->get_input_node_ptr(0));
        if (gather && ov::is_type<ops::Reshape>(gather->get_input_node_ptr(0)))
            return gather->get_input_node_ptr(0)->input_value(0);
    }
    return {};
}

Output<Node> control_input(const ops::Select& select) {
    const auto keep = ov::as_type<ops::NotEqual>(select.get_input_node_ptr(0));
    const auto index = keep ? ov::as_type<ops::Gather>(keep->get_input_node_ptr(0)) : nullptr;
    if (index && ov::is_type<ops::Reshape>(index->get_input_node_ptr(0)))
        return index->get_input_node_ptr(0)->input_value(0);
    return {};
}

Output<Node> reset_input(const ops::ScatterNDUpdate& write) {
    const auto reset_index = ov::as_type<ops::Unsqueeze>(write.get_input_node_ptr(0)->get_input_node_ptr(1));
    if (reset_index && ov::is_type<ops::Reshape>(reset_index->get_input_node_ptr(0)))
        return reset_index->get_input_node_ptr(0)->input_value(0);
    return {};
}

Output<Node> tensor_input(Output<Node> value, element::Type type, bool expanded = false) {
    if (expanded && ov::is_type<ops::Gather>(value.get_node())) {
        auto source = value.get_node()->input_value(0);
        if (rank_four(source) && source.get_partial_shape()[2] != value.get_partial_shape()[2])
            value = source;
    }
    if (type != element::f32 && ov::is_type<ops::Convert>(value.get_node()))
        value = value.get_node()->input_value(0);
    return value;
}

bool fuse_norm(OutputVector& args, float& q_eps, float& k_eps) {
    auto q = ov::as_type_ptr<ops::NormalizeL2>(args[0].get_node_shared_ptr());
    auto k = ov::as_type_ptr<ops::NormalizeL2>(args[1].get_node_shared_ptr());
    const auto valid = [](const std::shared_ptr<ops::NormalizeL2>& norm) {
        if (!norm || norm->get_output_element_type(0) != element::f32 || norm->get_eps_mode() != ov::op::EpsMode::ADD ||
            !std::isfinite(norm->get_eps()) || norm->get_eps() <= 0 ||
            !norm->get_control_dependencies().empty() || !norm->get_control_dependents().empty())
            return false;
        const auto axes = ov::as_type_ptr<ops::Constant>(norm->get_input_node_shared_ptr(1));
        if (!axes || ov::shape_size(axes->get_shape()) != 1)
            return false;
        const auto axis = axes->cast_vector<int64_t>()[0];
        return axis == -1 || axis == 3;
    };
    if (!valid(q) || !valid(k))
        return false;
    q_eps = q->get_eps();
    k_eps = k->get_eps();
    args[0] = q->input_value(0);
    args[1] = k->input_value(0);
    return true;
}

bool replay_matches(const OutputVector& outputs, const OutputVector& args, bool conv) {
    Match match;
    OutputVector reference_args;
    for (const auto& arg : args) {
        auto parameter = std::make_shared<ops::Parameter>(arg.get_element_type(), arg.get_partial_shape());
        reference_args.push_back(parameter);
        match.boundary.emplace(parameter.get(), arg);
    }

    const auto reference = ov::recurrent_replay::build(reference_args, conv);
    for (size_t i = 0; i < outputs.size(); ++i)
        if (!outputs_match(reference[i], outputs[i], match))
            return false;

    std::unordered_set<const Node*> region;
    for (const auto& [_, node] : match.nodes)
        region.insert(node);

    // No original-bank read may remain live after replacing the computation
    // and removing the Assign sinks. Check intermediate readers as well.
    OutputVector pending;
    for (size_t port : bank_ports(conv)) {
        if (std::count(args.begin(), args.end(), args[port]) != 1)
            return false;
        pending.push_back(args[port]);
    }
    std::unordered_set<const Node*> visited;
    for (size_t cursor = 0; cursor < pending.size(); ++cursor) {
        const auto value = pending[cursor];
        if (std::find(outputs.begin(), outputs.end(), value) != outputs.end())
            continue;
        for (const auto& reader : value.get_target_inputs()) {
            auto* node = reader.get_node();
            if (!visited.insert(node).second)
                continue;
            if (!region.count(node) || !node->get_control_dependents().empty() || !node->get_control_dependencies().empty())
                return false;
            for (const auto& output : node->outputs()) {
                if (std::find(args.begin(), args.end(), output) != args.end())
                    return false;
                pending.push_back(output);
            }
        }
    }
    return true;
}

void replace_replay(Model& model,
                    const Writebacks& writebacks,
                    const OutputVector& outputs,
                    const std::shared_ptr<Node>& fused,
                    bool conv) {
    fused->set_friendly_name(outputs[0].get_node()->get_friendly_name());
    ov::copy_runtime_info(outputs[0].get_node_shared_ptr(), fused);
    auto computation = outputs[0];
    computation.replace(fused->output(0));
    for (size_t i = 1; i < outputs.size(); ++i) {
        const auto& sink = writebacks.at(outputs[i].get_node());
        model.remove_sink(sink);
        model.remove_variable(sink->get_variable());
    }
    ov::disable_conversion(fused, element::f16);
    for (size_t port : bank_ports(conv)) {
        ov::mark_as_precision_sensitive(fused->input(port));
    }
}
template <typename Op>
Op* find_consumer(const Output<Node>& value) {
    for (const auto& target : value.get_target_inputs())
        if (target.get_index() == 0)
            if (const auto op = ov::as_type<Op>(target.get_node()))
                return op;
    return nullptr;
}

bool fuse_conv(Model& model,
               const Writebacks& writebacks,
               const std::unordered_set<const Node*>& live,
               const Output<Node>& bank_result,
               const ops::ScatterNDUpdate& write,
               const ops::Concat& update) {
    const auto history = ov::as_type<ops::Slice>(update.get_input_node_ptr(0));
    if (!history || !ov::is_type<ops::Concat>(history->get_input_node_ptr(0)))
        return false;
    const auto start = ov::as_type<ops::Unsqueeze>(history->get_input_node_ptr(1));
    const auto select = start ? ov::as_type<ops::Select>(start->get_input_node_ptr(0)) : nullptr;
    if (!select)
        return false;

    const auto columns = history->get_input_node_ptr(0);
    const auto layout = find_consumer<ops::Transpose>(history->input_value(0));
    const auto conv = layout ? find_consumer<ops::GroupConvolution>(layout->output(0)) : nullptr;
    const auto transposed = conv ? find_consumer<ops::Transpose>(conv->output(0)) : nullptr;
    const auto result = transposed ? find_consumer<ops::Unsqueeze>(transposed->output(0)) : nullptr;
    if (!result || !ov::is_type<ops::Reshape>(conv->get_input_node_ptr(1)))
        return false;

    // The columns contain the replayed window followed by the live input.
    auto x = tensor_input(columns->input_value(1), element::f16);
    if (!ov::is_type<ops::Squeeze>(x.get_node()))
        return false;
    x = x.get_node()->input_value(0);
    auto weights = tensor_input(conv->get_input_node_ptr(1)->input_value(0), element::f16);
    OutputVector args{bank_input(&write), rows_input(destination_rows(&write)), x, weights,
                      control_input(*select), reset_input(write)};
    if (!std::all_of(args.begin(), args.end(), rank_four))
        return false;
    const auto& weight_shape = weights.get_partial_shape();
    if (weight_shape[2].is_dynamic() || weight_shape[3].is_dynamic() || weight_shape[3].get_length() < 2)
        return false;

    Output<Node> convolution = result->output(0);
    if (x.get_element_type() != element::f32) {
        const auto cast = find_consumer<ops::Convert>(convolution);
        if (!cast)
            return false;
        convolution = cast->output(0);
    }
    if (std::none_of(args.begin(), args.end(), [](const Output<Node>& arg) {
            return arg.get_partial_shape().is_dynamic();
        }))
        return false;
    OutputVector outputs{convolution, bank_result};
    if (!live.count(convolution.get_node()) || !replay_matches(outputs, args, true))
        return false;
    auto fused = std::make_shared<op::RecurrentConvCacheReplayUpdate>(args, weight_shape[3].get_length() - 1);
    replace_replay(model, writebacks, outputs, fused, true);
    return true;
}
} // namespace

bool RecurrentReplayFusion::run_on_model(const std::shared_ptr<Model>& model) {
    const auto sinks = model->get_sinks();
    if (sinks.empty())
        return false;

    std::unordered_map<std::string, size_t> variable_users;
    for (const auto& node : model->get_ordered_ops())
        if (const auto* access = dynamic_cast<const ov::op::util::VariableExtension*>(node.get()))
            ++variable_users[access->get_variable_id()];

    // A fused writer must remain live after its Assign sinks are removed.
    // A consumer of only the output shape can let the GPU prune its computation.
    std::vector<const Node*> pending;
    for (const auto& result : model->get_results())
        pending.push_back(result.get());
    std::unordered_set<const Node*> live;
    while (!pending.empty()) {
        const auto* node = pending.back();
        pending.pop_back();
        if (ov::is_type<ov::op::util::ShapeOfBase>(node) || !live.insert(node).second)
            continue;
        for (size_t i = 0; i < node->get_input_size(); ++i)
            pending.push_back(node->get_input_node_ptr(i));
    }

    Writebacks writebacks;
    std::unordered_multimap<const ops::Slice*, Output<Node>> banks;
    for (const auto& sink : sinks) {
        const auto assign = ov::as_type_ptr<ops::Assign>(sink);
        if (!assign || !assign->get_control_dependencies().empty() || !assign->get_control_dependents().empty())
            continue;
        const auto value = assign->input_value(0);
        const auto write = bank_write(value);
        const auto bank = write ? bank_input(write) : Output<Node>{};
        if (!rank_four(bank) || !ov::is_type<ops::Parameter>(bank.get_node()) ||
            assign->get_variable_id() != replay_cache_state_prefix + bank.get_node()->get_friendly_name() ||
            variable_users[assign->get_variable_id()] != 1 || value.get_target_inputs().size() != 1)
            continue;
        if (const auto destination = destination_rows(write)) {
            writebacks.emplace(value.get_node(), assign);
            banks.emplace(destination, value);
        }
    }

    bool changed = false;
    for (const auto& [_, sink] : writebacks) {
        auto state_out = sink->input_value(0);
        const auto write = bank_write(state_out);
        if (!write || !rank_four(bank_input(write)))
            continue;
        auto updated = bank_update(write);
        if (!updated.get_node())
            continue;

        if (const auto update = ov::as_type<ops::Concat>(updated.get_node())) {
            try {
                changed |= fuse_conv(*model, writebacks, live, state_out, *write, *update);
            } catch (const ov::Exception&) {
                // Reject candidates with incompatible input shapes or types.
            }
            continue;
        }

        const auto select = ov::as_type_ptr<ops::Select>(updated.get_node_shared_ptr());
        if (!select)
            continue;
        auto loop = ov::as_type_ptr<ops::Loop>(select->get_input_node_shared_ptr(2));
        if (!loop || loop->get_input_size() != 8 || loop->get_output_size() != 2 || !loop->is_dynamic())
            continue;

        auto attention = loop->output(0);
        // Some producers request reduced-precision attention.
        for (const auto& target : attention.get_target_inputs()) {
            if (ov::is_type<ops::Convert>(target.get_node())) {
                const auto converted = target.get_node()->output(0);
                if (converted.get_element_type() != element::f32) {
                    attention = converted;
                    break;
                }
            }
        }

        if (!live.count(attention.get_node()))
            continue;

        // The loop inputs and bank writebacks identify the region's boundary.
        const auto type = attention.get_element_type();
        OutputVector args{
            tensor_input(loop->input_value(3), type, true), tensor_input(loop->input_value(4), type, true),
            tensor_input(loop->input_value(5), type), bank_input(write),
            tensor_input(loop->input_value(6), type), tensor_input(loop->input_value(7), type),
        };
        if (!std::all_of(args.begin(), args.end(), rank_four))
            continue;

        // Candidate pending-bank writebacks share the state bank's row/reset
        // metadata and store K, V, or the packed gate and beta prefix.
        const auto destination = destination_rows(write);
        if (!destination)
            continue;
        const auto first = banks.equal_range(destination);
        if (std::distance(first.first, first.second) != 4)
            continue;

        // The exact prefix source distinguishes the key and value banks
        // even when they have equal shapes or slot counts.
        const auto store_source = [type](const Output<Node>& bank) -> Output<Node> {
            const auto update = bank_update(bank_write(bank));
            const auto prefix = ov::as_type<ops::Concat>(update.get_node());
            if (!prefix || !ov::is_type<ops::Slice>(prefix->get_input_node_ptr(0)))
                return {};
            auto source = prefix->get_input_node_ptr(0)->input_value(0);
            return tensor_input(source, type);
        };
        OutputVector outputs{attention, state_out, {}, {}, {}};
        for (auto it = first.first; it != first.second; ++it) {
            const auto& bank = it->second;
            if (bank == state_out)
                continue;
            const auto source = store_source(bank);
            if (source == args[1] && source.get_node())
                outputs[2] = bank;
            else if (source == args[2] && source.get_node())
                outputs[3] = bank;
            else
                outputs[4] = bank;
        }
        if (!std::all_of(outputs.begin() + 2, outputs.end(), rank_four))
            continue;

        for (size_t i = 2; i < outputs.size(); ++i)
            args.push_back(bank_input(bank_write(outputs[i])));
        args.push_back(rows_input(destination));
        args.push_back(control_input(*select));
        args.push_back(reset_input(*write));
        if (!std::all_of(args.begin(), args.end(), rank_four))
            continue;

        try {
            const auto static_axis = [&](size_t port, size_t axis) {
                return args[port].get_partial_shape()[axis].is_static();
            };
            if (!static_axis(0, 2) || !static_axis(0, 3) || !static_axis(1, 2) || !static_axis(1, 3) ||
                !static_axis(2, 2) || !static_axis(2, 3) || !static_axis(4, 3) || !static_axis(6, 3) ||
                args[1].get_partial_shape()[2].get_length() == 0 || args[1].get_partial_shape()[3].get_length() == 0)
                continue;
            if (replay_matches(outputs, args, false)) {
                float q_eps = 1e-6f, k_eps = 1e-6f;
                const bool normalized = fuse_norm(args, q_eps, k_eps);
                auto fused = std::make_shared<op::RecurrentGatedDeltaNetReplayUpdate>(args, normalized, q_eps, k_eps);
                replace_replay(*model, writebacks, outputs, fused, false);
                changed = true;
            }
        } catch (const ov::Exception&) {
            // The candidate has a similar writeback layout but different tensor shapes.
        }
    }
    return changed;
}
} // namespace ov::intel_gpu
