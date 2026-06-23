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

namespace {

using NodeToGroupMapPtr = ov::npuw::online::detail::OVNodeToGroupMapPtr;

void isolate_node(const std::shared_ptr<ov::Node>& node,
                  const std::string& isol_tag,
                  const NodeToGroupMapPtr& node_to_gptr) {
    if (node && node_to_gptr->count(node)) {
        node_to_gptr->at(node)->isolate(isol_tag);
    }
}

// Scan output_multiply's middle dimensions (all dims except dim-0=num_experts and last=hidden).
// If any middle dim > 1 it is the token count (prefill). If all are 1, it is decoding.
// Shape layouts:  GPT-OSS [N, token, 1, H]  /  Qwen3 [N, 1, token, H]
bool is_decoding_stage(const std::shared_ptr<ov::Node>& output_multiply) {
    const auto shape = output_multiply->get_output_partial_shape(0);
    if (!shape.rank().is_static()) {
        return false;
    }
    const auto rank = shape.rank().get_length();
    for (int64_t i = 1; i < rank - 1; ++i) {
        const auto& d = shape[i];
        if (d.is_static() && d.get_length() != 1) {
            return false;  // prefill: found token dim > 1
        }
    }
    return true;  // all middle dims are 1 → decoding
}

// Find the first consumer of `node` satisfying `pred`. Returns nullptr if not found.
template <typename Pred>
std::shared_ptr<ov::Node> find_consumer_by_type(const std::shared_ptr<ov::Node>& node, Pred&& pred) {
    for (auto& output : node->outputs()) {
        for (auto& input : output.get_target_inputs()) {
            auto consumer = input.get_node()->shared_from_this();
            if (pred(consumer)) {
                return consumer;
            }
        }
    }
    return nullptr;
}

// After output_multiply, find the first ReduceSum consumer and isolate it.
void isolate_reduce_sum_after(const std::shared_ptr<ov::Node>& output_multiply,
                              const std::string& isol_tag,
                              const NodeToGroupMapPtr& node_to_gptr) {
    std::shared_ptr<ov::Node> reduce_sum;
    for (auto& out : output_multiply->outputs()) {
        for (auto& in : out.get_target_inputs()) {
            auto consumer = in.get_node()->shared_from_this();
            if (std::dynamic_pointer_cast<ov::op::v1::ReduceSum>(consumer)) {
                reduce_sum = consumer;
                break;
            }
        }
        if (reduce_sum)
            break;
    }
    if (reduce_sum && node_to_gptr->count(reduce_sum)) {
        node_to_gptr->at(reduce_sum)->isolate(isol_tag);
        LOG_DEBUG("  ReduceSum successfully isolated");
    } else if (reduce_sum) {
        LOG_WARN("  ReduceSum found but not in node_to_gptr map");
    } else {
        LOG_WARN("  No ReduceSum found after Multiply (unexpected for decoding stage)");
    }
}

}  // namespace

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

        // Check if optional AWQ multiply was matched
        auto matched_swish = node_to_output.at(swish).get_node_shared_ptr();
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

        auto matched_output_multiply = node_to_output.at(output_multiply).get_node_shared_ptr();

        LOG_DEBUG("Expert Multiply output_shape: " << matched_output_multiply->get_output_partial_shape(0));
        const bool is_decoding = is_decoding_stage(matched_output_multiply);
        LOG_DEBUG("GPT-OSS Expert pattern matched (" << (is_decoding ? "Decoding" : "Prefill") << " stage)");

        auto isolate = [&](const std::shared_ptr<ov::Node>& pattern_node) {
            isolate_node(node_to_output.at(pattern_node).get_node_shared_ptr(), isol_tag, node_to_gptr);
        };

        isolate(tile);
        isolate(reshape1);
        isolate(weights_multiply1);
        isolate(weights_convert1);
        isolate(matmul1);
        isolate(add1);
        isolate(slice);
        isolate(minimum);
        isolate_node(matched_swish, isol_tag, node_to_gptr);

        // Isolate AWQ multiply if it exists
        if (matched_awq_multiply) {
            isolate_node(matched_awq_multiply, isol_tag, node_to_gptr);
            LOG_DEBUG("AWQ multiply after Swish isolated");
        }

        isolate(other_slice);
        isolate(clamp);
        isolate(add2);
        isolate(multiply1);
        isolate(weights_multiply2);
        isolate(weights_convert2);
        isolate(matmul2);
        isolate(add3);
        isolate(reshape2);
        isolate_node(matched_output_multiply, isol_tag, node_to_gptr);

        if (is_decoding) {
            LOG_DEBUG("Decoding stage detected, searching for ReduceSum to isolate...");
            isolate_reduce_sum_after(matched_output_multiply, isol_tag, node_to_gptr);
        }

        return false;
    };

    register_matcher(std::make_shared<opp::Matcher>(output_multiply, "TagGPTOSSExpert"), std::move(callback));
}

/*
    GPT-OSS Router Pattern (constant-folded variant):

    All ShapeOf nodes in the router are constant-folded before this pattern runs.
    The resulting graph is:

        weights -> Multiply -> Convert -> MatMul -> Add -> TopK
                                                          |\-> output(0) values -> Softmax -> Slice
                                                          \-> output(1) indices -> Convert (indices)

    Slice -> ScatterElementsUpdate -> Transpose -> Reshape -> Unsqueeze
    Broadcast (zero base for Scatter, shape fed by folded Const)

    Pattern-matched (6): Multiply, Convert, MatMul, Add, TopK, Softmax, Slice
    Manually retrieved (4): topk_convert (indices Convert), Broadcast, Scatter,
                            Transpose, Reshape, Unsqueeze
*/
GPTOSSRouter::GPTOSSRouter(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag) {
    LOG_DEBUG("GPTOSSRouter pattern matcher registered with tag: " << isol_tag);

    // Pattern-matched nodes (7): Multiply, Convert, MatMul, Add, TopK, Softmax, Slice
    // topk_convert (indices Convert) is retrieved manually - TopK has two outputs and
    // wrap_type always binds output(0), so output(1)->Convert cannot be expressed inline.
    auto weights_multiply = opp::wrap_type<ov::op::v1::Multiply>({opp::any_input(), opp::any_input()});
    auto weights_convert2 = opp::wrap_type<ov::op::v0::Convert>({weights_multiply});
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), weights_convert2});
    auto add = opp::wrap_type<ov::op::v1::Add>({matmul, opp::any_input()});
    auto topk = opp::wrap_type<ov::op::v11::TopK>({add, opp::any_input()});
    auto softmax = opp::wrap_type<ov::op::v8::Softmax>({topk});  // connects to topk->output(0)

    // Pattern root: Slice data input is Softmax output; all shape inputs are any_input()
    // because ShapeOf nodes are constant-folded before this pattern runs.
    auto slice = opp::wrap_type<ov::op::v8::Slice>(
        {softmax, opp::any_input(), opp::any_input(), opp::any_input(), opp::any_input()});

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

        // Get pattern-matched nodes needed for manual retrieval
        auto matched_slice = node_to_output.at(slice).get_node_shared_ptr();

        // topk_convert: Convert on TopK indices output (output(1)).  Not in the formal
        // pattern because TopK has two outputs and wrap_type always binds output(0).
        std::shared_ptr<ov::Node> matched_topk_convert = nullptr;
        for (auto& target : matched_topk->output(1).get_target_inputs()) {
            auto consumer = target.get_node()->shared_from_this();
            if (std::dynamic_pointer_cast<ov::op::v0::Convert>(consumer)) {
                matched_topk_convert = consumer;
                break;
            }
        }

        // Manual retrieval (7 nodes): helper function
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
        auto isolate = [&](const std::shared_ptr<ov::Node>& pattern_node) {
            isolate_node(node_to_output.at(pattern_node).get_node_shared_ptr(), isol_tag, node_to_gptr);
        };

        isolate(weights_multiply);
        isolate(weights_convert2);
        isolate(matmul);
        isolate(add);
        isolate_node(matched_topk, isol_tag, node_to_gptr);
        isolate(softmax);
        isolate_node(matched_topk_convert, isol_tag, node_to_gptr);
        isolate_node(matched_slice, isol_tag, node_to_gptr);
        isolate_node(matched_broadcast, isol_tag, node_to_gptr);
        isolate_node(matched_scatter, isol_tag, node_to_gptr);
        isolate_node(matched_transpose, isol_tag, node_to_gptr);
        isolate_node(matched_reshape, isol_tag, node_to_gptr);
        isolate_node(matched_unsqueeze, isol_tag, node_to_gptr);

        LOG_DEBUG("Router pattern isolated");
        return false;
    };

    register_matcher(std::make_shared<opp::Matcher>(slice, "TagGPTOSSRouter"), std::move(callback));
}

/*
    Qwen3 Expert Pattern:

    Input preparation:
        Tile -> Reshape1

    Gate projection (SwiGLU gate branch):
        Reshape1 -> MatMul_gate (with weights: Multiply -> Convert) -> Swish

    Up projection (SwiGLU up branch):
        Reshape1 -> MatMul_up  (with weights: Multiply -> Convert)

    SwiGLU merge:
        Swish + MatMul_up -> Multiply_swiglu

    Down projection:
        Multiply_swiglu -> MatMul_down (with weights: Multiply -> Convert) -> Reshape2

    Output (scaled by router scores):
        Reshape2 * router_score -> Multiply_output   <-- pattern root
        (router_score = opp::any_input(), produced entirely by Qwen3Router)

    Isolation boundary:
    - Expert claims: Tile, Reshape1, gate/up/down MatMuls+weights, SwiGLU Multiply, Reshape2, output Multiply
    - Router claims: Softmax, TopK, ReduceSum, Divide, ScatterElementsUpdate, Transpose, Reshape_score, Unsqueeze_score
    - Shared shape-compute nodes (ShapeOf->Gather->Unsqueeze->Concat chains) stay outside both,
      becoming parameter inputs at subgraph boundaries.
*/
Qwen3Expert::Qwen3Expert(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag) {
    LOG_DEBUG("Qwen3Expert pattern matcher registered with tag: " << isol_tag);

    // Input preparation: Tile -> Reshape
    auto tile = opp::wrap_type<ov::op::v0::Tile>({opp::any_input(), opp::any_input()});
    auto reshape1 = opp::wrap_type<ov::op::v1::Reshape>({tile, opp::any_input()});

    // Gate projection weights: Multiply(quantized weight, scale) -> Convert
    auto gate_weights_multiply = opp::wrap_type<ov::op::v1::Multiply>({opp::any_input(), opp::any_input()});
    auto gate_weights_convert = opp::wrap_type<ov::op::v0::Convert>({gate_weights_multiply});
    // Gate MatMul + Swish activation
    auto matmul_gate = opp::wrap_type<ov::op::v0::MatMul>({reshape1, gate_weights_convert});
    auto swish = opp::wrap_type<ov::op::v4::Swish>({matmul_gate});

    // Up projection weights: Multiply(quantized weight, scale) -> Convert
    auto up_weights_multiply = opp::wrap_type<ov::op::v1::Multiply>({opp::any_input(), opp::any_input()});
    auto up_weights_convert = opp::wrap_type<ov::op::v0::Convert>({up_weights_multiply});
    // Up MatMul
    auto matmul_up = opp::wrap_type<ov::op::v0::MatMul>({reshape1, up_weights_convert});

    // SwiGLU: gate * up
    auto multiply_swiglu = opp::wrap_type<ov::op::v1::Multiply>({swish, matmul_up});

    // Down projection weights: Multiply(quantized weight, scale) -> Convert
    auto down_weights_multiply = opp::wrap_type<ov::op::v1::Multiply>({opp::any_input(), opp::any_input()});
    auto down_weights_convert = opp::wrap_type<ov::op::v0::Convert>({down_weights_multiply});
    // Down MatMul -> Reshape
    auto matmul_down = opp::wrap_type<ov::op::v0::MatMul>({multiply_swiglu, down_weights_convert});
    auto reshape2 = opp::wrap_type<ov::op::v1::Reshape>({matmul_down, opp::any_input()});

    // Pattern root: expert_output * router_score
    // The router score (Unsqueeze output) is produced entirely by Qwen3Router and flows
    // in as opp::any_input() here to avoid double-claiming shared nodes.
    auto output_multiply = opp::wrap_type<ov::op::v1::Multiply>({reshape2, opp::any_input()});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        auto matched_tile = node_to_output.at(tile).get_node_shared_ptr();
        auto matched_output_multiply = node_to_output.at(output_multiply).get_node_shared_ptr();

        LOG_DEBUG("Qwen3Expert pattern matched: " << matched_tile->get_friendly_name());

        LOG_DEBUG("Qwen3 Expert Multiply output_shape: " << matched_output_multiply->get_output_partial_shape(0));
        const bool is_decoding = is_decoding_stage(matched_output_multiply);
        LOG_DEBUG("Qwen3 Expert pattern matched (" << (is_decoding ? "Decoding" : "Prefill") << " stage)");

        auto isolate = [&](const std::shared_ptr<ov::Node>& pattern_node) {
            isolate_node(node_to_output.at(pattern_node).get_node_shared_ptr(), isol_tag, node_to_gptr);
        };

        isolate(tile);
        isolate(reshape1);
        isolate(gate_weights_multiply);
        isolate(gate_weights_convert);
        isolate(matmul_gate);
        isolate(swish);
        isolate(up_weights_multiply);
        isolate(up_weights_convert);
        isolate(matmul_up);
        isolate(multiply_swiglu);
        isolate(down_weights_multiply);
        isolate(down_weights_convert);
        isolate(matmul_down);
        isolate(reshape2);
        isolate_node(matched_output_multiply, isol_tag, node_to_gptr);

        if (is_decoding) {
            LOG_DEBUG("Decoding stage detected, searching for ReduceSum to isolate...");
            isolate_reduce_sum_after(matched_output_multiply, isol_tag, node_to_gptr);
        }

        return false;
    };

    register_matcher(std::make_shared<opp::Matcher>(output_multiply, "TagQwen3Expert"), std::move(callback));
}

/*
    Qwen3 Router Pattern:

    Router weights (quantized, dequantized via weight chain):
        Convert(weight) -> Multiply(weight, scale) -> Convert -> MatMul(input, weight)

    Score computation:
        MatMul -> Softmax -> TopK(values, indices)

    Score normalization:
        TopK(values) -> ReduceSum -> Divide(values, sum)   [renormalize over K selected]

    Scatter to full expert dimension:
        TopK(indices) + Divide(scores) -> ScatterElementsUpdate(zero_broadcast, indices, scores)

    Shape to [num_experts, token_count, 1, 1] for expert broadcast:
        ScatterElementsUpdate -> Transpose -> Reshape -> Unsqueeze   <-- pattern root

    Note: The Unsqueeze output is consumed by Qwen3Expert's Multiply_output node.
    Key difference from GPT-OSS: Softmax is BEFORE TopK (not after),
    requiring explicit renormalization via ReduceSum->Divide.
*/
Qwen3Router::Qwen3Router(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag) {
    LOG_DEBUG("Qwen3Router pattern matcher registered with tag: " << isol_tag);

    // Router weights: Convert(weight) -> Multiply(weight, scale) -> Convert -> MatMul
    auto weights_convert_in = opp::wrap_type<ov::op::v0::Convert>({opp::any_input()});
    auto weights_multiply = opp::wrap_type<ov::op::v1::Multiply>({weights_convert_in, opp::any_input()});
    auto weights_convert_out = opp::wrap_type<ov::op::v0::Convert>({weights_multiply});
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), weights_convert_out});

    // Score: Softmax -> TopK
    auto softmax = opp::wrap_type<ov::op::v8::Softmax>({matmul});
    auto topk = opp::wrap_type<ov::op::v11::TopK>({softmax, opp::any_input()});

    // Renormalization: TopK(values)->ReduceSum, TopK(values)/ReduceSum = Divide
    auto reduce_sum = opp::wrap_type<ov::op::v1::ReduceSum>({topk, opp::any_input()});
    auto divide = opp::wrap_type<ov::op::v1::Divide>({topk, reduce_sum});

    // Scatter to full expert shape (pattern root = Unsqueeze)
    auto scatter =
        opp::wrap_type<ov::op::v12::ScatterElementsUpdate>({opp::any_input(), topk, divide, opp::any_input()});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({scatter, opp::any_input()});
    auto reshape = opp::wrap_type<ov::op::v1::Reshape>({transpose, opp::any_input()});
    auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({reshape, opp::any_input()});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();

        // Validate: TopK should be MAX mode (selecting top-K experts)
        auto matched_topk = node_to_output.at(topk).get_node_shared_ptr();
        auto topk_node = std::dynamic_pointer_cast<ov::op::v11::TopK>(matched_topk);
        if (!topk_node || topk_node->get_mode() != ov::op::v11::TopK::Mode::MAX) {
            return false;
        }

        LOG_DEBUG("Qwen3Router pattern matched: " << matched_topk->get_friendly_name());

        auto matched_scatter = node_to_output.at(scatter).get_node_shared_ptr();

        // Also isolate Broadcast node that provides zero-filled base for ScatterElementsUpdate
        auto broadcast_node = matched_scatter->input_value(0).get_node_shared_ptr();
        auto matched_broadcast = std::dynamic_pointer_cast<ov::op::v3::Broadcast>(broadcast_node);

        auto isolate = [&](const std::shared_ptr<ov::Node>& pattern_node) {
            isolate_node(node_to_output.at(pattern_node).get_node_shared_ptr(), isol_tag, node_to_gptr);
        };

        isolate(weights_convert_in);
        isolate(weights_multiply);
        isolate(weights_convert_out);
        isolate(matmul);
        isolate(softmax);
        isolate_node(matched_topk, isol_tag, node_to_gptr);
        isolate(reduce_sum);
        isolate(divide);
        isolate_node(matched_broadcast, isol_tag, node_to_gptr);
        isolate_node(matched_scatter, isol_tag, node_to_gptr);
        isolate(transpose);
        isolate(reshape);
        isolate(unsqueeze);

        return false;
    };

    register_matcher(std::make_shared<opp::Matcher>(unsqueeze, "TagQwen3Router"), std::move(callback));
}

}  // namespace moe
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
