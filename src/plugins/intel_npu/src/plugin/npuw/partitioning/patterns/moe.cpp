// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe.hpp"

#include "../../logging.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace npuw {
namespace patterns {
namespace moe {

namespace opp = ov::pass::pattern;

/*
    GPT-OSS Expert Pattern:

    Input:
        Tile -> Reshape1

    First MatMul (gate + up projections):
        Reshape1 -> MatMul1 (with weights Convert1) -> Add1

    Dual branches from Add1:

        Gate branch (left):
            Add1 -> Slice2 -> Clamp -> Add2 (with Convert constant)  ──┐
                                                                       │
        Activation branch (right):                                     │
            Add1 -> Slice1 -> Minimum (with Convert constant)          │
                           -> Swish (with const input) ────────────────┤
                                                                       │
    Merge branches:                                                    │
            Add2 + Swish -> Multiply1  <───────────────────────────────┘

    Second MatMul (down projection):
        Multiply1 -> MatMul2 (with weights Convert2) -> Add3

    Output (stage-dependent):
        Prefill stage:  Add3 -> Reshape2 -> Multiply (output here)
                        ReduceSum will be performed in downstream subgraph

        Decoding stage: Add3 -> Reshape2 -> Multiply -> ReduceSum (output here)
                        ReduceSum is included in expert subgraph

    Isolation strategy:
    - Prefill (token_count > 1): Isolate up to Multiply, ReduceSum stays in downstream
    - Decoding (token_count == 1): Isolate including ReduceSum, self-contained expert subgraph
*/
GPTOSSExpert::GPTOSSExpert(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag) {
    LOG_DEBUG("GPTOSSExpert pattern matcher registered with tag: " << isol_tag);

    // Input preparation
    auto tile = opp::wrap_type<ov::op::v0::Tile>({opp::any_input(), opp::any_input()});
    auto reshape1 = opp::wrap_type<ov::op::v1::Reshape>({tile, opp::any_input()});

    // First MatMul (gate + up projections) - weights path: Multiply -> Convert -> MatMul
    auto weights_multiply1 = opp::wrap_type<ov::op::v1::Multiply>({opp::any_input(), opp::any_input()});
    auto weights_convert1 = opp::wrap_type<ov::op::v0::Convert>({weights_multiply1});
    auto matmul1 = opp::wrap_type<ov::op::v0::MatMul>({reshape1, weights_convert1});
    auto add1 = opp::wrap_type<ov::op::v1::Add>({matmul1, opp::any_input()});

    // Activation branch: Slice -> Minimum -> Swish -> (optional AWQ Multiply)
    auto slice = opp::wrap_type<ov::op::v8::Slice>(
        {add1, opp::any_input(), opp::any_input(), opp::any_input(), opp::any_input()});
    auto minimum = opp::wrap_type<ov::op::v1::Minimum>({slice, opp::any_input()});
    auto swish = opp::wrap_type<ov::op::v4::Swish>({minimum, opp::any_input()});

    // AWQ quantization may add an optional Multiply after Swish
    auto awq_multiply = opp::optional<ov::op::v1::Multiply>({swish, opp::any_input()});

    // Gate branch: Slice -> Clamp -> Add2
    auto other_slice = opp::wrap_type<ov::op::v8::Slice>(
        {add1, opp::any_input(), opp::any_input(), opp::any_input(), opp::any_input()});
    auto clamp = opp::wrap_type<ov::op::v0::Clamp>({other_slice});
    auto add2 = opp::wrap_type<ov::op::v1::Add>({clamp, opp::any_input()});

    // Merge branches - awq_multiply will be Swish if not matched, or AWQ Multiply if matched
    auto multiply1 = opp::wrap_type<ov::op::v1::Multiply>({add2, awq_multiply});

    // Second MatMul (down projection) - weights path: Multiply -> Convert -> MatMul
    auto weights_multiply2 = opp::wrap_type<ov::op::v1::Multiply>({opp::any_input(), opp::any_input()});
    auto weights_convert2 = opp::wrap_type<ov::op::v0::Convert>({weights_multiply2});
    auto matmul2 = opp::wrap_type<ov::op::v0::MatMul>({multiply1, weights_convert2});
    auto add3 = opp::wrap_type<ov::op::v1::Add>({matmul2, opp::any_input()});

    // Output reshape - Multiply
    auto reshape2 = opp::wrap_type<ov::op::v1::Reshape>({add3, opp::any_input()});
    auto output_multiply = opp::wrap_type<ov::op::v1::Multiply>({reshape2, opp::any_input()});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_tile = node_to_output.at(tile).get_node_shared_ptr();
        auto matched_reshape1 = node_to_output.at(reshape1).get_node_shared_ptr();
        auto matched_weights_multiply1 = node_to_output.at(weights_multiply1).get_node_shared_ptr();
        auto matched_weights_convert1 = node_to_output.at(weights_convert1).get_node_shared_ptr();
        auto matched_matmul1 = node_to_output.at(matmul1).get_node_shared_ptr();
        auto matched_add1 = node_to_output.at(add1).get_node_shared_ptr();
        auto matched_slice = node_to_output.at(slice).get_node_shared_ptr();
        auto matched_minimum = node_to_output.at(minimum).get_node_shared_ptr();
        auto matched_swish = node_to_output.at(swish).get_node_shared_ptr();

        // Check if optional AWQ multiply was matched
        std::shared_ptr<ov::Node> matched_awq_multiply = nullptr;
        if (node_to_output.count(awq_multiply) > 0) {
            auto awq_multiply_value = node_to_output.at(awq_multiply);
            if (awq_multiply_value.get_node_shared_ptr() != matched_swish) {
                // AWQ multiply exists and is different from swish
                matched_awq_multiply = awq_multiply_value.get_node_shared_ptr();
                LOG_DEBUG("AWQ multiply detected after Swish: " << matched_awq_multiply->get_friendly_name());
            }
        } else {
            LOG_DEBUG("Normal model: No AWQ multiply after Swish");
        }

        auto matched_other_slice = node_to_output.at(other_slice).get_node_shared_ptr();
        auto matched_clamp = node_to_output.at(clamp).get_node_shared_ptr();
        auto matched_add2 = node_to_output.at(add2).get_node_shared_ptr();
        auto matched_multiply1 = node_to_output.at(multiply1).get_node_shared_ptr();
        auto matched_weights_multiply2 = node_to_output.at(weights_multiply2).get_node_shared_ptr();
        auto matched_weights_convert2 = node_to_output.at(weights_convert2).get_node_shared_ptr();
        auto matched_matmul2 = node_to_output.at(matmul2).get_node_shared_ptr();
        auto matched_add3 = node_to_output.at(add3).get_node_shared_ptr();
        auto matched_reshape2 = node_to_output.at(reshape2).get_node_shared_ptr();
        auto matched_output_multiply = node_to_output.at(output_multiply).get_node_shared_ptr();

        // Check if this is decoding stage by examining shape[rank-2]
        auto output_shape = matched_output_multiply->get_output_partial_shape(0);
        LOG_DEBUG("Expert Multiply output_shape: " << output_shape);
        bool is_decoding = false;

        if (output_shape.rank().is_static() && output_shape.rank().get_length() >= 2) {
            auto rank = output_shape.rank().get_length();
            auto token_dim = output_shape[rank - 2];

            if (token_dim.is_static() && token_dim.get_length() == 1) {
                is_decoding = true;
                LOG_DEBUG("GPT-OSS Expert pattern matched (Decoding stage): single token");
            } else if (token_dim.is_static()) {
                LOG_DEBUG("GPT-OSS Expert pattern matched (Prefill stage): token_count=" << token_dim.get_length());
            }
        }

        // Isolate all common expert nodes
        node_to_gptr->at(matched_tile)->isolate(isol_tag);
        node_to_gptr->at(matched_reshape1)->isolate(isol_tag);
        node_to_gptr->at(matched_weights_multiply1)->isolate(isol_tag);
        node_to_gptr->at(matched_weights_convert1)->isolate(isol_tag);
        node_to_gptr->at(matched_matmul1)->isolate(isol_tag);
        node_to_gptr->at(matched_add1)->isolate(isol_tag);
        node_to_gptr->at(matched_slice)->isolate(isol_tag);
        node_to_gptr->at(matched_minimum)->isolate(isol_tag);
        node_to_gptr->at(matched_swish)->isolate(isol_tag);

        // Isolate AWQ multiply if it exists
        if (matched_awq_multiply && node_to_gptr->count(matched_awq_multiply)) {
            node_to_gptr->at(matched_awq_multiply)->isolate(isol_tag);
            LOG_DEBUG("AWQ multiply after Swish isolated");
        }

        node_to_gptr->at(matched_other_slice)->isolate(isol_tag);
        node_to_gptr->at(matched_clamp)->isolate(isol_tag);
        node_to_gptr->at(matched_add2)->isolate(isol_tag);
        node_to_gptr->at(matched_multiply1)->isolate(isol_tag);
        node_to_gptr->at(matched_weights_multiply2)->isolate(isol_tag);
        node_to_gptr->at(matched_weights_convert2)->isolate(isol_tag);
        node_to_gptr->at(matched_matmul2)->isolate(isol_tag);
        node_to_gptr->at(matched_add3)->isolate(isol_tag);
        node_to_gptr->at(matched_reshape2)->isolate(isol_tag);
        node_to_gptr->at(matched_output_multiply)->isolate(isol_tag);

        // If decoding stage, find and isolate ReduceSum after Multiply
        if (is_decoding) {
            LOG_DEBUG("Decoding stage detected, searching for ReduceSum to isolate...");
            std::shared_ptr<ov::Node> matched_reduce_sum = nullptr;

            for (auto& output : matched_output_multiply->outputs()) {
                for (auto& input : output.get_target_inputs()) {
                    auto consumer = input.get_node()->shared_from_this();
                    if (auto reduce_sum = std::dynamic_pointer_cast<ov::op::v1::ReduceSum>(consumer)) {
                        matched_reduce_sum = reduce_sum;
                        LOG_DEBUG("  Found ReduceSum after Multiply, isolating for decoding stage");
                        break;
                    }
                }
                if (matched_reduce_sum)
                    break;
            }

            if (matched_reduce_sum && node_to_gptr->count(matched_reduce_sum)) {
                node_to_gptr->at(matched_reduce_sum)->isolate(isol_tag);
                LOG_DEBUG("  ReduceSum successfully isolated");
            } else if (matched_reduce_sum) {
                LOG_WARN("  ReduceSum found but not in node_to_gptr map");
            } else {
                LOG_WARN("  No ReduceSum found after Multiply (unexpected for decoding stage)");
            }
        }

        return false;
    };

    register_matcher(std::make_shared<opp::Matcher>(output_multiply, "TagGPTOSSExpert"), std::move(callback));
}

/*
    GPT-OSS Router Pattern:

    Matches MoE router layer (16 nodes: 9 pattern-matched + 7 manual retrieval)
    Pattern: weights -> Multiply -> Convert -> MatMul -> Add -> TopK -> Softmax -> Slice
                                                             \-> Convert -> ShapeOf -/
    Manual: Add -> ShapeOf -> Broadcast -> Scatter -> Transpose -> Reshape -> Unsqueeze
*/
GPTOSSRouter::GPTOSSRouter(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag) {
    LOG_DEBUG("GPTOSSRouter pattern matcher registered with tag: " << isol_tag);

    // Pattern-matched nodes (9 total)
    auto weights_multiply = opp::wrap_type<ov::op::v1::Multiply>({opp::any_input(), opp::any_input()});
    auto weights_convert2 = opp::wrap_type<ov::op::v0::Convert>({weights_multiply});
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), weights_convert2});
    auto add = opp::wrap_type<ov::op::v1::Add>({matmul, opp::any_input()});
    auto topk = opp::wrap_type<ov::op::v11::TopK>({add, opp::any_input()});
    auto softmax = opp::wrap_type<ov::op::v8::Softmax>({topk});
    auto topk_convert = opp::wrap_type<ov::op::v0::Convert>({topk});
    auto shapeof_topk = opp::wrap_type<ov::op::v3::ShapeOf>({topk_convert});

    // Pattern root
    auto slice = opp::wrap_type<ov::op::v8::Slice>(
        {softmax, opp::any_input(), shapeof_topk, opp::any_input(), opp::any_input()});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        // Validate TopK mode (MAX) and Router keywords
        auto matched_topk = node_to_output.at(topk).get_node_shared_ptr();
        auto topk_node = std::dynamic_pointer_cast<ov::op::v11::TopK>(matched_topk);
        if (!topk_node || topk_node->get_mode() != ov::op::v11::TopK::Mode::MAX) {
            return false;
        }

        std::string topk_name = matched_topk->get_friendly_name();
        // Check if node name contains MoE router/expert patterns
        bool is_router = (topk_name.find(MLP_ROUTER_NAME) != std::string::npos ||
                          topk_name.find(MLP_EXPERT_NAME) != std::string::npos);
        if (!is_router) {
            return false;
        }

        LOG_DEBUG("GPT-OSS Router pattern matched: " << topk_name);

        // Get pattern-matched nodes
        auto matched_weights_multiply = node_to_output.at(weights_multiply).get_node_shared_ptr();
        auto matched_weights_convert2 = node_to_output.at(weights_convert2).get_node_shared_ptr();
        auto matched_matmul = node_to_output.at(matmul).get_node_shared_ptr();
        auto matched_add = node_to_output.at(add).get_node_shared_ptr();
        auto matched_softmax = node_to_output.at(softmax).get_node_shared_ptr();
        auto matched_topk_convert = node_to_output.at(topk_convert).get_node_shared_ptr();
        auto matched_shapeof_topk = node_to_output.at(shapeof_topk).get_node_shared_ptr();
        auto matched_slice = node_to_output.at(slice).get_node_shared_ptr();

        // Manual retrieval (7 nodes): helper function
        auto find_consumer_by_type =
            [](const std::shared_ptr<ov::Node>& node,
               const std::function<bool(const std::shared_ptr<ov::Node>&)>& pred) -> std::shared_ptr<ov::Node> {
            for (auto& output : node->outputs()) {
                for (auto& input : output.get_target_inputs()) {
                    auto consumer = input.get_node()->shared_from_this();
                    if (pred(consumer)) {
                        return consumer;
                    }
                }
            }
            return nullptr;
        };

        auto matched_scatter = find_consumer_by_type(matched_slice, [](const std::shared_ptr<ov::Node>& n) {
            return std::dynamic_pointer_cast<ov::op::v3::ScatterElementsUpdate>(n) ||
                   std::dynamic_pointer_cast<ov::op::v12::ScatterElementsUpdate>(n);
        });
        if (!matched_scatter) {
            LOG_DEBUG("Router pattern: ScatterElementsUpdate not found");
            return false;
        }

        // Retrieve Broadcast and ShapeOf
        auto broadcast_node = matched_scatter->input_value(0).get_node_shared_ptr();
        auto matched_broadcast = std::dynamic_pointer_cast<ov::op::v3::Broadcast>(broadcast_node);
        if (!matched_broadcast) {
            LOG_DEBUG("Router pattern: Broadcast not found");
            return false;
        }

        std::shared_ptr<ov::Node> matched_shapeof = nullptr;
        for (size_t i = 0; i < matched_broadcast->inputs().size(); ++i) {
            auto input_node = matched_broadcast->input_value(i).get_node_shared_ptr();
            if (std::dynamic_pointer_cast<ov::op::v3::ShapeOf>(input_node)) {
                matched_shapeof = input_node;
                break;
            }
        }
        if (!matched_shapeof) {
            LOG_DEBUG("Router pattern: ShapeOf not found");
            return false;
        }

        // Retrieve output chain (Transpose -> Reshape -> Unsqueeze)
        auto matched_transpose = find_consumer_by_type(matched_scatter, [](const std::shared_ptr<ov::Node>& n) {
            return std::dynamic_pointer_cast<ov::op::v1::Transpose>(n) != nullptr;
        });
        if (!matched_transpose) {
            LOG_DEBUG("Router pattern: Transpose not found");
            return false;
        }

        auto matched_reshape = find_consumer_by_type(matched_transpose, [](const std::shared_ptr<ov::Node>& n) {
            return std::dynamic_pointer_cast<ov::op::v1::Reshape>(n) != nullptr;
        });
        if (!matched_reshape) {
            LOG_DEBUG("Router pattern: Reshape not found");
            return false;
        }

        auto matched_unsqueeze = find_consumer_by_type(matched_reshape, [](const std::shared_ptr<ov::Node>& n) {
            return std::dynamic_pointer_cast<ov::op::v0::Unsqueeze>(n) != nullptr;
        });
        if (!matched_unsqueeze) {
            LOG_DEBUG("Router pattern: Unsqueeze not found");
            return false;
        }

        // Isolate all 16 nodes
        auto isolate_if_exists = [&](const std::shared_ptr<ov::Node>& node) {
            if (node && node_to_gptr->count(node)) {
                node_to_gptr->at(node)->isolate(isol_tag);
            }
        };

        isolate_if_exists(matched_weights_multiply);
        isolate_if_exists(matched_weights_convert2);
        isolate_if_exists(matched_matmul);
        isolate_if_exists(matched_add);
        isolate_if_exists(matched_topk);
        isolate_if_exists(matched_softmax);
        isolate_if_exists(matched_topk_convert);
        isolate_if_exists(matched_shapeof_topk);
        isolate_if_exists(matched_slice);
        isolate_if_exists(matched_shapeof);
        isolate_if_exists(matched_broadcast);
        isolate_if_exists(matched_scatter);
        isolate_if_exists(matched_transpose);
        isolate_if_exists(matched_reshape);
        isolate_if_exists(matched_unsqueeze);

        LOG_DEBUG("Router pattern isolated");
        return false;
    };

    register_matcher(std::make_shared<opp::Matcher>(slice, "TagGPTOSSRouter"), std::move(callback));
}

}  // namespace moe
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
