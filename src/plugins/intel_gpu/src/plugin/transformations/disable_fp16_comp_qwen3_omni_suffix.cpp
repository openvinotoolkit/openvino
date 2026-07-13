// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_comp_qwen3_omni_suffix.hpp"

#include <deque>
#include <optional>
#include <unordered_set>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/core/rt_info.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_gpu {

namespace {

struct GatedMLPBlock {
    std::shared_ptr<ov::Node> source;
    std::shared_ptr<ov::op::v1::Add> residual_add;
    std::shared_ptr<ov::Node> residual_input;
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

    return GatedMLPBlock{gate->input_value(0).get_node_shared_ptr(), residual_add, residual_input};
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

std::shared_ptr<ov::Node> find_sensitive_suffix_boundary(const std::shared_ptr<ov::Model>& model) {
    size_t read_values = 0;
    size_t assigns = 0;
    size_t attentions = 0;
    std::vector<GatedMLPBlock> blocks;

    for (const auto& node : model->get_ordered_ops()) {
        read_values += ov::is_type<ov::op::v6::ReadValue>(node);
        assigns += ov::is_type<ov::op::v6::Assign>(node);
        attentions += ov::is_type<ov::op::v13::ScaledDotProductAttention>(node);
        if (auto block = match_gated_mlp_block(node))
            blocks.push_back(*block);
    }

    if (read_values != 10 || assigns != 10 || attentions != 5 || blocks.size() != 5 ||
        !has_code_predictor_output_contract(model)) {
        return nullptr;
    }

    // Reject five unrelated MLPs. Each block source must be downstream of the
    // preceding block's residual output, forming one transformer stack.
    for (size_t i = 1; i < blocks.size(); ++i) {
        if (!depends_on(blocks[i].source, blocks[i - 1].residual_add.get()))
            return nullptr;
    }

    // The measured safe boundary is the residual input of the third MLP. It
    // is the post-attention residual; the third MLP itself belongs to suffix.
    return blocks[2].residual_input;
}

}  // namespace

bool DisableFP16CompForQwen3OmniCodePredictorSuffix::run_on_model(const std::shared_ptr<ov::Model>& model) {
    auto boundary = find_sensitive_suffix_boundary(model);
    if (!boundary)
        return false;

    std::deque<std::shared_ptr<ov::Node>> pending;
    for (const auto& output : boundary->outputs()) {
        for (const auto& input : output.get_target_inputs())
            pending.push_back(input.get_node()->shared_from_this());
    }

    std::unordered_set<const ov::Node*> visited;
    std::vector<std::shared_ptr<ov::op::v6::Assign>> cache_writes;
    std::vector<std::shared_ptr<ov::Node>> protected_nodes;
    bool changed = false;

    while (!pending.empty()) {
        auto node = pending.front();
        pending.pop_front();
        if (!node || !visited.insert(node.get()).second)
            continue;

        if (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(node)) {
            cache_writes.push_back(assign);
            // Assign is a state sink. Do not follow its state/control edge
            // back to ReadValue, which belongs to the next inference step.
            continue;
        } else if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(node)) {
            // SDPA is decomposed after this pass and does not propagate the
            // precision marker to its internal Transpose/scale operations.
            // Keep the native f16 SDPA path and protect its surrounding
            // projections/residual computation instead.
        } else if (!ov::is_type<ov::op::v0::Result>(node)) {
            // ConvertPrecision consults this attribute before lowering f32 to
            // f16. Integer-only nodes are harmlessly unaffected.
            ov::disable_conversion(node, ov::element::f16);
            protected_nodes.push_back(node);
            changed = true;
        }

        for (const auto& output : node->outputs()) {
            for (const auto& input : output.get_target_inputs())
                pending.push_back(input.get_node()->shared_from_this());
        }
    }

    // Constants can be shared by early FP16 layers and the protected suffix.
    // Put an explicit f16 -> f32 barrier on each protected edge. The f16
    // midpoint prevents f32 requirements from propagating to the shared
    // source. If constant-source folding evaluates the barrier, it produces a
    // suffix-private value and still cannot affect an early consumer.
    for (const auto& node : protected_nodes) {
        for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node->input_value(input_idx).get_node_shared_ptr());
            if (!constant || !constant->get_output_element_type(0).is_real() ||
                constant->output(0).get_target_inputs().size() < 2) {
                continue;
            }
            auto to_f16 = std::make_shared<ov::op::v0::Convert>(constant, ov::element::f16);
            auto to_f32 = std::make_shared<ov::op::v0::Convert>(to_f16, ov::element::f32);
            to_f16->set_friendly_name(constant->get_friendly_name() + "/to_f16_boundary");
            to_f32->set_friendly_name(constant->get_friendly_name() + "/to_fp32_suffix");
            ov::copy_runtime_info(constant, {to_f16, to_f32});
            ov::disable_conversion(to_f32, ov::element::f16);
            node->input(input_idx).replace_source_output(to_f32->output(0));
            changed = true;
        }
    }

    // At this point Variables and Assign inputs are f32. ConvertPrecision
    // later changes both Variable types and these unprotected Converts to f16,
    // while the producer side remains f32. This avoids an f32 Assign input /
    // f16 Variable type mismatch and keeps cache storage compact.
    for (const auto& assign : cache_writes) {
        auto cache_convert = std::make_shared<ov::op::v0::Convert>(assign->input_value(0), ov::element::f32);
        cache_convert->set_friendly_name(assign->get_friendly_name() + "/to_fp16_cache");
        assign->input(0).replace_source_output(cache_convert);
        changed = true;
    }

    return changed;
}

}  // namespace ov::intel_gpu