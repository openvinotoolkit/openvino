// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_comp_qwen3_omni_suffix.hpp"

#include <deque>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/swish.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_gpu {
namespace {

struct GatedMLPBlock {
    std::shared_ptr<ov::Node> source;
    std::shared_ptr<ov::op::v1::Add> residual_add;
    std::shared_ptr<ov::Node> residual_input;
    std::vector<std::shared_ptr<ov::op::v0::MatMul>> input_projections;
};

struct SensitiveMatch {
    std::shared_ptr<ov::Node> suffix_boundary;
    std::vector<std::shared_ptr<ov::op::v13::ScaledDotProductAttention>> attentions;
    std::unordered_set<const ov::Node*> fp16_mlp_input_projections;
};

std::optional<GatedMLPBlock> match_gated_mlp_block(const std::shared_ptr<ov::Node>& node) {
    auto down = ov::as_type_ptr<ov::op::v0::MatMul>(node);
    if (!down)
        return std::nullopt;

    std::shared_ptr<ov::op::v1::Multiply> product;
    for (const auto& input : down->input_values()) {
        product = ov::as_type_ptr<ov::op::v1::Multiply>(input.get_node_shared_ptr());
        if (product)
            break;
    }
    if (!product)
        return std::nullopt;

    std::shared_ptr<ov::op::v4::Swish> swish;
    std::shared_ptr<ov::op::v0::MatMul> up;
    for (const auto& input : product->input_values()) {
        auto input_node = input.get_node_shared_ptr();
        if (auto candidate = ov::as_type_ptr<ov::op::v4::Swish>(input_node))
            swish = candidate;
        else if (auto candidate = ov::as_type_ptr<ov::op::v0::MatMul>(input_node))
            up = candidate;
    }
    if (!swish || !up)
        return std::nullopt;

    auto gate = ov::as_type_ptr<ov::op::v0::MatMul>(swish->input_value(0).get_node_shared_ptr());
    if (!gate || gate->input_value(0) != up->input_value(0))
        return std::nullopt;

    std::shared_ptr<ov::op::v1::Add> residual_add;
    for (const auto& target : down->output(0).get_target_inputs()) {
        auto candidate = ov::as_type_ptr<ov::op::v1::Add>(target.get_node()->shared_from_this());
        if (!candidate)
            continue;
        if (residual_add)
            return std::nullopt;
        residual_add = candidate;
    }
    if (!residual_add)
        return std::nullopt;

    std::shared_ptr<ov::Node> residual_input;
    for (const auto& input : residual_add->input_values()) {
        if (input.get_node() != down.get()) {
            if (residual_input)
                return std::nullopt;
            residual_input = input.get_node_shared_ptr();
        }
    }
    if (!residual_input)
        return std::nullopt;

    return GatedMLPBlock{gate->input_value(0).get_node_shared_ptr(), residual_add, residual_input, {gate, up}};
}

bool depends_on(const std::shared_ptr<ov::Node>& node, const ov::Node* ancestor) {
    std::deque<std::shared_ptr<ov::Node>> pending{node};
    std::unordered_set<const ov::Node*> visited;
    while (!pending.empty()) {
        auto current = pending.front();
        pending.pop_front();
        if (!current || !visited.insert(current.get()).second)
            continue;
        if (current.get() == ancestor)
            return true;
        for (const auto& input : current->input_values())
            pending.push_back(input.get_node_shared_ptr());
    }
    return false;
}

std::vector<std::shared_ptr<ov::op::v6::ReadValue>> collect_read_values(const ov::Output<ov::Node>& output) {
    std::deque<std::shared_ptr<ov::Node>> pending{output.get_node_shared_ptr()};
    std::unordered_set<const ov::Node*> visited;
    std::vector<std::shared_ptr<ov::op::v6::ReadValue>> read_values;
    while (!pending.empty()) {
        auto current = pending.front();
        pending.pop_front();
        if (!current || !visited.insert(current.get()).second)
            continue;
        if (auto read_value = ov::as_type_ptr<ov::op::v6::ReadValue>(current))
            read_values.push_back(read_value);
        for (const auto& input : current->input_values())
            pending.push_back(input.get_node_shared_ptr());
    }
    return read_values;
}

std::vector<std::shared_ptr<ov::op::v6::ReadValue>> difference(
    const std::vector<std::shared_ptr<ov::op::v6::ReadValue>>& values,
    const std::vector<std::shared_ptr<ov::op::v6::ReadValue>>& excluded) {
    std::unordered_set<const ov::Node*> excluded_nodes;
    for (const auto& node : excluded)
        excluded_nodes.insert(node.get());

    std::vector<std::shared_ptr<ov::op::v6::ReadValue>> result;
    for (const auto& node : values) {
        if (!excluded_nodes.count(node.get()))
            result.push_back(node);
    }
    return result;
}

bool is_stateful_attention(const std::shared_ptr<ov::op::v13::ScaledDotProductAttention>& attention,
                           const std::unordered_set<std::string>& assigned_variables) {
    if (attention->get_input_size() < 3)
        return false;

    const auto query_states = collect_read_values(attention->input_value(0));
    const auto key_states = difference(collect_read_values(attention->input_value(1)), query_states);
    auto value_states = difference(collect_read_values(attention->input_value(2)), query_states);
    if (key_states.size() != 1)
        return false;
    value_states = difference(value_states, key_states);
    if (value_states.size() != 1)
        return false;

    return assigned_variables.count(key_states.front()->get_variable_id()) &&
           assigned_variables.count(value_states.front()->get_variable_id());
}

bool find_direct_attention(const std::shared_ptr<ov::Node>& branch,
                           const std::unordered_set<std::string>& assigned_variables,
                           std::shared_ptr<ov::op::v13::ScaledDotProductAttention>& attention) {
    std::deque<std::shared_ptr<ov::Node>> pending{branch};
    std::unordered_set<const ov::Node*> visited;
    attention.reset();
    while (!pending.empty()) {
        auto current = pending.front();
        pending.pop_front();
        if (!current || !visited.insert(current.get()).second)
            continue;
        if (auto candidate = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(current)) {
            if (attention && attention != candidate)
                return false;
            attention = candidate;
            continue;
        }
        for (const auto& input : current->input_values())
            pending.push_back(input.get_node_shared_ptr());
    }
    return !attention || is_stateful_attention(attention, assigned_variables);
}

bool has_code_predictor_output_contract(const std::shared_ptr<ov::Model>& model) {
    if (model->get_results().size() != 2)
        return false;

    bool has_code = false;
    bool has_embedding = false;
    for (const auto& result : model->get_results()) {
        const auto output = result->input_value(0);
        const auto& shape = output.get_partial_shape();
        if (!shape.rank().is_static())
            return false;
        if (output.get_element_type() == ov::element::i64 && shape.rank().get_length() == 2 &&
            shape[1].compatible(1)) {
            has_code = true;
        } else if (output.get_element_type() == ov::element::f32 && shape.rank().get_length() == 3 &&
                   shape[1].compatible(1) && shape[2].compatible(1024)) {
            has_embedding = true;
        }
    }
    return has_code && has_embedding;
}

std::optional<SensitiveMatch> find_sensitive_regions(const std::shared_ptr<ov::Model>& model) {
    std::vector<GatedMLPBlock> blocks;
    std::unordered_set<std::string> assigned_variables;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(node))
            assigned_variables.insert(assign->get_variable_id());
        if (auto block = match_gated_mlp_block(node))
            blocks.push_back(*block);
    }
    if (!has_code_predictor_output_contract(model))
        return std::nullopt;

    struct BlockLink {
        bool valid = false;
        std::optional<size_t> predecessor;
        std::shared_ptr<ov::op::v13::ScaledDotProductAttention> attention;
    };
    std::vector<BlockLink> links(blocks.size());
    std::vector<std::vector<size_t>> successors(blocks.size());

    for (size_t i = 0; i < blocks.size(); ++i) {
        auto attention_residual = ov::as_type_ptr<ov::op::v1::Add>(blocks[i].residual_input);
        if (!attention_residual || attention_residual->get_input_size() != 2 ||
            !depends_on(blocks[i].source, attention_residual.get()))
            continue;

        std::optional<size_t> predecessor;
        bool ambiguous_predecessor = false;
        size_t predecessor_input = 0;
        for (size_t input_idx = 0; input_idx < attention_residual->get_input_size(); ++input_idx) {
            const auto* input_node = attention_residual->input_value(input_idx).get_node();
            for (size_t candidate = 0; candidate < blocks.size(); ++candidate) {
                if (input_node != blocks[candidate].residual_add.get())
                    continue;
                if (predecessor && *predecessor != candidate)
                    ambiguous_predecessor = true;
                else
                    predecessor = candidate;
                predecessor_input = input_idx;
            }
        }
        if (ambiguous_predecessor)
            continue;

        std::shared_ptr<ov::op::v13::ScaledDotProductAttention> attention;
        if (predecessor) {
            const auto branch = attention_residual->input_value(1 - predecessor_input).get_node_shared_ptr();
            if (!find_direct_attention(branch, assigned_variables, attention) || !attention)
                continue;
        } else {
            std::shared_ptr<ov::op::v13::ScaledDotProductAttention> first;
            std::shared_ptr<ov::op::v13::ScaledDotProductAttention> second;
            if (!find_direct_attention(attention_residual->input_value(0).get_node_shared_ptr(),
                                       assigned_variables,
                                       first) ||
                !find_direct_attention(attention_residual->input_value(1).get_node_shared_ptr(),
                                       assigned_variables,
                                       second) ||
                static_cast<bool>(first) == static_cast<bool>(second))
                continue;
            attention = first ? first : second;
        }

        links[i] = BlockLink{true, predecessor, attention};
        if (predecessor)
            successors[*predecessor].push_back(i);
    }

    std::optional<SensitiveMatch> match;
    for (size_t candidate = 0; candidate < blocks.size(); ++candidate) {
        if (!links[candidate].valid || !links[candidate].predecessor || successors[candidate].size() != 1)
            continue;
        const auto previous = *links[candidate].predecessor;
        if (!links[previous].valid || !links[previous].predecessor || successors[previous].size() != 1)
            continue;
        const auto first = *links[previous].predecessor;
        if (!links[first].valid || links[first].predecessor || successors[first].size() != 1)
            continue;
        const auto next = successors[candidate].front();
        if (!links[next].valid || successors[next].size() != 1)
            continue;
        const auto last = successors[next].front();
        if (!links[last].valid || !successors[last].empty())
            continue;

        bool feeds_all_results = true;
        for (const auto& result : model->get_results()) {
            if (!depends_on(result->input_value(0).get_node_shared_ptr(), blocks[last].residual_add.get())) {
                feeds_all_results = false;
                break;
            }
        }
        if (!feeds_all_results)
            continue;
        if (match)
            return std::nullopt;

        std::unordered_set<const ov::Node*> fp16_mlp_input_projections;
        for (const auto block_idx : {candidate, next, last}) {
            for (const auto& projection : blocks[block_idx].input_projections)
                fp16_mlp_input_projections.insert(projection.get());
        }
        match = SensitiveMatch{blocks[candidate].residual_input,
                               {links[first].attention,
                                links[previous].attention,
                                links[candidate].attention,
                                links[next].attention,
                                links[last].attention},
                               std::move(fp16_mlp_input_projections)};
    }
    return match;
}

void protect_attention_input(const ov::Output<ov::Node>& output, bool protect_projection) {
    std::deque<std::shared_ptr<ov::Node>> pending{output.get_node_shared_ptr()};
    std::unordered_set<const ov::Node*> visited;
    while (!pending.empty()) {
        auto node = pending.front();
        pending.pop_front();
        if (!node || !visited.insert(node.get()).second)
            continue;
        if (ov::is_type<ov::op::v0::MatMul>(node)) {
            if (protect_projection)
                ov::disable_conversion(node, ov::element::f16);
            continue;
        }
        ov::disable_conversion(node, ov::element::f16);
        if (ov::is_type<ov::op::v6::ReadValue>(node))
            continue;
        for (const auto& input : node->input_values())
            pending.push_back(input.get_node_shared_ptr());
    }
}

}  // namespace

static bool mark_qwen3_omni_code_predictor_precision(const std::shared_ptr<ov::Model>& model) {
    const auto match = find_sensitive_regions(model);
    if (!match || !match->suffix_boundary || match->attentions.size() != 5)
        return false;

    // Preserve SDPA arithmetic, the recurrent KV state, and all non-projection
    // Q/K/V frontier operations. Layer 0 projections stay FP32; layers 1-2 use
    // FP16; layers 3-4 are kept FP32 by the downstream suffix.
    std::unordered_set<std::string> protected_variable_ids;
    for (size_t layer = 0; layer < match->attentions.size(); ++layer) {
        const auto& attention = match->attentions[layer];
        ov::disable_conversion(attention, ov::element::f16);
        for (size_t input_idx = 0; input_idx < 3; ++input_idx) {
            // Layer 0 projections remain FP32. Layers 1-2 use FP16, while
            // layers 3-4 are protected again by the downstream suffix.
            protect_attention_input(attention->input_value(input_idx), layer == 0);
            for (const auto& state : collect_read_values(attention->input_value(input_idx)))
                protected_variable_ids.insert(state->get_variable_id());
        }
    }
    for (const auto& node : model->get_ordered_ops()) {
        if (auto read = ov::as_type_ptr<ov::op::v6::ReadValue>(node)) {
            if (protected_variable_ids.count(read->get_variable_id())) {
                ov::disable_conversion(read, ov::element::f16);
            }
        } else if (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(node)) {
            if (protected_variable_ids.count(assign->get_variable_id())) {
                ov::disable_conversion(assign, ov::element::f16);
            }
        }
    }

    std::deque<std::shared_ptr<ov::Node>> pending;
    for (const auto& output : match->suffix_boundary->outputs()) {
        for (const auto& input : output.get_target_inputs())
            pending.push_back(input.get_node()->shared_from_this());
    }

    std::unordered_set<const ov::Node*> visited;
    while (!pending.empty()) {
        auto node = pending.front();
        pending.pop_front();
        if (!node || !visited.insert(node.get()).second)
            continue;
        if (ov::is_type<ov::op::v6::Assign>(node)) {
            continue;
        } else if (!ov::is_type<ov::op::v0::Result>(node) &&
                   !match->fp16_mlp_input_projections.count(node.get())) {
            ov::disable_conversion(node, ov::element::f16);
        }
        for (const auto& output : node->outputs()) {
            for (const auto& input : output.get_target_inputs())
                pending.push_back(input.get_node()->shared_from_this());
        }
    }
    return true;
}

ConvertPrecisionForQwen3OmniCodePredictor::ConvertPrecisionForQwen3OmniCodePredictor(
    const precisions_map& precisions)
    : m_precisions(precisions) {}

bool ConvertPrecisionForQwen3OmniCodePredictor::run_on_model(const std::shared_ptr<ov::Model>& model) {
    const bool is_qwen3_omni_code_predictor = mark_qwen3_omni_code_predictor_precision(model);
    ov::pass::ConvertPrecision convert_precision(m_precisions,
                                                 {},
                                                 true,
                                                 false,
                                                 !is_qwen3_omni_code_predictor);
    convert_precision.set_pass_config(get_pass_config());
    return convert_precision.run_on_model(model);
}

}  // namespace ov::intel_gpu
