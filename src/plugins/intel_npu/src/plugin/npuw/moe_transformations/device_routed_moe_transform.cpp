// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "device_routed_moe_transform.hpp"

#include <optional>
#include <queue>
#include <unordered_set>

#include "../logging.hpp"
#include "moe_transformation_utils.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/ops.hpp"

namespace ov {
namespace npuw {
namespace pass {

namespace {

// ============================================================================
// Helper structures for organizing transformation data
// ============================================================================

struct LayerNodes {
    std::vector<std::shared_ptr<ov::op::v0::Tile>> tiles;
    std::vector<std::shared_ptr<ov::op::v1::Reshape>> constant_reshapes;
    std::vector<std::shared_ptr<ov::op::v1::Reshape>> dynamic_reshapes;
    std::vector<std::shared_ptr<ov::op::v0::MatMul>> matmuls;
    std::vector<std::shared_ptr<ov::op::v1::Add>> adds;
    std::vector<std::shared_ptr<ov::op::v1::Multiply>> multiplies;
    std::shared_ptr<ov::op::v1::Transpose> transpose;
    size_t num_experts = 0;

    bool has_required_nodes() const {
        return (!constant_reshapes.empty() || !dynamic_reshapes.empty()) && (!matmuls.empty() || !adds.empty());
    }
};

struct RouterInfo {
    std::shared_ptr<ov::op::v11::TopK> topk_node;
    ov::Output<ov::Node> topk_indices_raw;
    ov::Output<ov::Node> router_scores;  // scores from Scatter.input(2), replaces Scatter->Transpose
    int64_t k_value;
    std::shared_ptr<ov::op::v1::Transpose> scatter_transpose;  // the Scatter -> Transpose node
    std::shared_ptr<ov::op::v1::Multiply> output_multiply;     // expert x scores Multiply
};

// ============================================================================
// Helper functions
// ============================================================================

// Trace back through Convert to find actual weight/bias source
inline ov::Output<ov::Node> get_weight_source(const ov::Output<ov::Node>& input) {
    auto node = input.get_node_shared_ptr();
    if (auto convert = std::dynamic_pointer_cast<ov::op::v0::Convert>(node)) {
        return convert->input_value(0);
    }
    return input;
}

// Check if a reshape operation is unsqueeze-like (only inserts dimensions with size 1).
// Used in collect_from_expert_output to detect activation-path Reshapes whose shape
// is not a Constant node (e.g. driven by ShapeOf chains) but are safe to replace with
// an Unsqueeze of the data input.
bool is_unsqueeze_like_reshape(const std::shared_ptr<ov::op::v1::Reshape>& reshape) {
    auto input_shape = reshape->input_value(0).get_partial_shape();
    auto output_shape = reshape->get_output_partial_shape(0);

    // Must have static ranks and output rank must be greater than input rank
    if (!input_shape.rank().is_static() || !output_shape.rank().is_static() ||
        output_shape.rank().get_length() <= input_shape.rank().get_length()) {
        return false;
    }

    // Verify all output dims are either from input or newly inserted with size 1
    int64_t in_idx = 0;
    for (int64_t out_idx = 0; out_idx < output_shape.rank().get_length(); ++out_idx) {
        if (in_idx < input_shape.rank().get_length() &&
            (!input_shape[in_idx].is_static() || !output_shape[out_idx].is_static() ||
             input_shape[in_idx].get_length() == output_shape[out_idx].get_length())) {
            // This dimension matches input dimension
            ++in_idx;
        } else if (!output_shape[out_idx].is_static() || output_shape[out_idx].get_length() != 1) {
            // Not a size-1 inserted dimension - not unsqueeze-like
            return false;
        }
        // else: this is a newly inserted dimension with size 1, continue
    }

    // Valid unsqueeze if all input dimensions were matched
    return in_idx == input_shape.rank().get_length();
}

// ============================================================================
// Router detection by topology
// ============================================================================

// Detects a MoE router pattern starting from a ScatterElementsUpdate node.
// Expected downstream chain: Scatter -> Transpose -> [Reshape/Unsqueeze]* -> Multiply -> ReduceSum
// The Scatter.input(1) must trace back to a TopK (router top-k selection).
// This approach is name-independent and works for both GPT-OSS and Qwen3.
std::optional<RouterInfo> detect_router_by_topology(const std::shared_ptr<ov::Node>& scatter) {
    // 1. Find Transpose as a direct consumer of Scatter.output(0)
    std::shared_ptr<ov::op::v1::Transpose> transpose;
    for (const auto& ti : scatter->output(0).get_target_inputs()) {
        if (auto t = std::dynamic_pointer_cast<ov::op::v1::Transpose>(ti.get_node()->shared_from_this())) {
            transpose = t;
            break;
        }
    }
    if (!transpose)
        return std::nullopt;

    // 2. Walk downstream: Transpose -> [Reshape/Unsqueeze]* -> Multiply
    // Follow single-consumer chain to find the output_multiply (expert x scores)
    auto follow_single = [](const std::shared_ptr<ov::Node>& n) -> std::shared_ptr<ov::Node> {
        const auto targets = n->output(0).get_target_inputs();
        if (targets.size() != 1)
            return nullptr;
        return targets.begin()->get_node()->shared_from_this();
    };

    std::shared_ptr<ov::op::v1::Multiply> output_multiply;
    auto cur = follow_single(transpose);
    for (int hops = 0; hops < 4 && cur; ++hops) {
        if (auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(cur)) {
            output_multiply = mul;
            break;
        }
        if (!std::dynamic_pointer_cast<ov::op::v1::Reshape>(cur) &&
            !std::dynamic_pointer_cast<ov::op::v0::Unsqueeze>(cur)) {
            return std::nullopt;
        }
        cur = follow_single(cur);
    }
    if (!output_multiply)
        return std::nullopt;

    // 3. Verify ReduceSum as the single consumer of output_multiply
    if (!std::dynamic_pointer_cast<ov::op::v1::ReduceSum>(follow_single(output_multiply)))
        return std::nullopt;

    // 4. Trace Scatter.input(1) back to TopK (indices, possibly through Convert)
    auto indices_node = scatter->input_value(1).get_node_shared_ptr();
    if (auto conv = std::dynamic_pointer_cast<ov::op::v0::Convert>(indices_node)) {
        indices_node = conv->input_value(0).get_node_shared_ptr();
    }
    auto topk_node = std::dynamic_pointer_cast<ov::op::v11::TopK>(indices_node);
    if (!topk_node || topk_node->get_mode() != ov::op::v11::TopK::Mode::MAX) {
        return std::nullopt;
    }

    // 5. Validate TopK indices shape: batch dimension must be 1 (decoding stage only)
    auto topk_indices_raw = topk_node->output(1);
    auto indices_shape = topk_indices_raw.get_partial_shape();
    if (indices_shape.rank().is_static() && indices_shape.rank().get_length() >= 1) {
        if (indices_shape[0].is_static() && indices_shape[0].get_length() != 1) {
            return std::nullopt;
        }
    }

    // 6. Extract K value from TopK's constant K input
    auto k_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(topk_node->input_value(1).get_node_shared_ptr());
    if (!k_const)
        return std::nullopt;
    int64_t k_value = k_const->cast_vector<int64_t>()[0];

    // 7. Router scores = Scatter.input(2): the k scores that were scattered into position
    //    GPT-OSS: Slice(Softmax(TopK.out(0)))   Qwen3: Divide(TopK.out(0), ReduceSum(...))
    auto router_scores = scatter->input_value(2);

    LOG_INFO("DeviceRoutedMoE: Detected MoE router by topology, K=" << k_value);

    return RouterInfo{topk_node, topk_indices_raw, router_scores, k_value, transpose, output_multiply};
}

// ============================================================================
// Expert node collection by backward BFS from expert output
// ============================================================================

// Collects expert subgraph nodes by tracing backward from the expert computation output.
// output_multiply has two inputs: expert computation output (BFS entry) and router scores (skip).
// This approach is name-independent and works for both GPT-OSS and Qwen3.
LayerNodes collect_from_expert_output(const RouterInfo& router) {
    LayerNodes nodes;
    nodes.transpose = router.scatter_transpose;

    // Identify which input of output_multiply is the expert computation output.
    // The other input comes from the Scatter->Transpose chain (router broadcast path).
    auto is_from_scatter_transpose = [&](const std::shared_ptr<ov::Node>& n) -> bool {
        std::shared_ptr<ov::Node> cur = n;
        for (int hops = 0; hops < 5 && cur; ++hops) {
            if (cur == router.scatter_transpose)
                return true;
            // Follow input(0) through shape ops (Reshape has data at input 0, Unsqueeze too)
            if (std::dynamic_pointer_cast<ov::op::v1::Reshape>(cur) ||
                std::dynamic_pointer_cast<ov::op::v0::Unsqueeze>(cur)) {
                cur = cur->input_value(0).get_node_shared_ptr();
            } else {
                break;
            }
        }
        return false;
    };

    ov::Output<ov::Node> expert_output;
    bool found = false;
    for (size_t i = 0; i < 2 && !found; ++i) {
        auto inp = router.output_multiply->input_value(i);
        if (!is_from_scatter_transpose(inp.get_node_shared_ptr())) {
            expert_output = inp;
            found = true;
        }
    }
    if (!found) {
        LOG_WARN("collect_from_expert_output: cannot identify expert output in output_multiply");
        return nodes;
    }

    // Phase 1: BFS backward from expert_output, collecting all activation-path nodes.
    // Constant-derived inputs (weights, scales, biases) are skipped in BFS but collected in phase 3.
    std::queue<std::shared_ptr<ov::Node>> queue;
    std::unordered_set<std::shared_ptr<ov::Node>> visited;

    auto start = expert_output.get_node_shared_ptr();
    queue.push(start);
    visited.insert(start);

    std::vector<std::shared_ptr<ov::op::v0::Tile>> all_tiles;
    std::vector<std::shared_ptr<ov::op::v1::Reshape>> all_const_reshapes;
    std::vector<std::shared_ptr<ov::op::v1::Reshape>> all_dyn_reshapes;
    std::vector<std::shared_ptr<ov::op::v0::MatMul>> all_matmuls;
    std::vector<std::shared_ptr<ov::op::v1::Add>> all_adds;
    std::vector<std::shared_ptr<ov::op::v1::Multiply>> all_multiplies;

    while (!queue.empty()) {
        auto n = queue.front();
        queue.pop();

        if (auto tile = std::dynamic_pointer_cast<ov::op::v0::Tile>(n)) {
            all_tiles.push_back(tile);
            continue;  // Tile is the boundary; don't BFS further into shared hidden state
        }
        if (auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(n)) {
            auto shape_src = reshape->input_value(1).get_node_shared_ptr();
            if (std::dynamic_pointer_cast<ov::op::v0::Constant>(shape_src)) {
                all_const_reshapes.push_back(reshape);
            } else if (is_unsqueeze_like_reshape(reshape)) {
                all_dyn_reshapes.push_back(reshape);
            }
        } else if (auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(n)) {
            all_matmuls.push_back(matmul);
        } else if (auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(n)) {
            all_adds.push_back(add);
        } else if (auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(n)) {
            all_multiplies.push_back(mul);
        }

        // Enqueue non-constant-derived inputs (activation paths only)
        for (size_t i = 0; i < n->get_input_size(); ++i) {
            auto inp_node = n->input_value(i).get_node_shared_ptr();
            if (visited.count(inp_node))
                continue;
            if (moe_utils::is_constant_derived(inp_node))
                continue;
            if (std::dynamic_pointer_cast<ov::op::v0::Parameter>(inp_node))
                continue;
            visited.insert(inp_node);
            queue.push(inp_node);
        }
    }

    // Phase 2: Determine num_experts from Tile repeats[0] > k_value
    for (auto& tile : all_tiles) {
        auto repeats_const =
            std::dynamic_pointer_cast<ov::op::v0::Constant>(tile->input_value(1).get_node_shared_ptr());
        if (!repeats_const)
            continue;
        auto rdata = repeats_const->cast_vector<int64_t>();
        if (!rdata.empty() && rdata[0] > router.k_value) {
            if (nodes.num_experts == 0) {
                nodes.num_experts = static_cast<size_t>(rdata[0]);
            }
            nodes.tiles.push_back(tile);
        }
    }

    if (nodes.num_experts == 0) {
        LOG_WARN("collect_from_expert_output: no Tile with repeats[0] > k_value found");
        return nodes;
    }

    // Phase 3: Filter collected nodes — keep only those whose weight/shape uses the expert dimension
    for (auto& reshape : all_const_reshapes) {
        auto shape_const =
            std::dynamic_pointer_cast<ov::op::v0::Constant>(reshape->input_value(1).get_node_shared_ptr());
        auto shape_data = shape_const->cast_vector<int64_t>();
        if (!shape_data.empty() && shape_data[0] == static_cast<int64_t>(nodes.num_experts)) {
            nodes.constant_reshapes.push_back(reshape);
        }
    }

    nodes.dynamic_reshapes = std::move(all_dyn_reshapes);

    // Returns true if a constant somewhere in the weight chain has shape[0] == expert_dim.
    // Recursively peels Convert and both inputs of Multiply (the two recognized chain patterns).
    // Returns false for any unrecognized node type, signalling an unknown weight chain.
    std::function<bool(const ov::Output<ov::Node>&, size_t, bool&)> weight_has_expert_dim;
    weight_has_expert_dim = [&](const ov::Output<ov::Node>& input, size_t expert_dim, bool& chain_recognized) -> bool {
        auto node = input.get_node_shared_ptr();
        if (auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(node)) {
            return !c->get_shape().empty() && c->get_shape()[0] == expert_dim;
        }
        if (auto conv = std::dynamic_pointer_cast<ov::op::v0::Convert>(node)) {
            return weight_has_expert_dim(conv->input_value(0), expert_dim, chain_recognized);
        }
        if (auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(node)) {
            return weight_has_expert_dim(mul->input_value(0), expert_dim, chain_recognized) ||
                   weight_has_expert_dim(mul->input_value(1), expert_dim, chain_recognized);
        }
        // Unrecognized node in weight chain (e.g. Reshape, Gather, custom op).
        chain_recognized = false;
        return false;
    };

    for (auto& matmul : all_matmuls) {
        bool chain_recognized = true;
        bool has_expert_dim = weight_has_expert_dim(matmul->input_value(1), nodes.num_experts, chain_recognized);
        if (!has_expert_dim) {
            continue;  // weight doesn't use expert dimension — not an expert MatMul
        }
        if (!chain_recognized) {
            // Weight has the expert dimension but through an unrecognized chain: we cannot safely
            // insert a Gather into it. Transforming other nodes (Tile, Reshape) without transforming
            // this MatMul's weight would produce a shape mismatch at inference time.
            LOG_WARN("collect_from_expert_output: MatMul '" << matmul->get_friendly_name()
                                                            << "' has expert-dim weight through an unrecognized chain; "
                                                               "skipping transformation for this layer");
            nodes.matmuls.clear();
            return nodes;
        }
        nodes.matmuls.push_back(matmul);
    }

    for (auto& add : all_adds) {
        for (size_t i = 0; i < 2; ++i) {
            auto bias_source = get_weight_source(add->input_value(i));
            if (auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(bias_source.get_node_shared_ptr())) {
                if (!c->get_shape().empty() && c->get_shape()[0] == nodes.num_experts) {
                    nodes.adds.push_back(add);
                    break;
                }
            }
        }
    }

    for (auto& mul : all_multiplies) {
        // Skip if used as MatMul weight input (quantized weight chain, handled by transform_matmuls)
        bool used_by_matmul = false;
        for (const auto& out : mul->outputs()) {
            for (const auto& ti : out.get_target_inputs()) {
                if (std::dynamic_pointer_cast<ov::op::v0::MatMul>(ti.get_node()->shared_from_this())) {
                    used_by_matmul = true;
                    break;
                }
            }
            if (used_by_matmul)
                break;
        }
        if (used_by_matmul)
            continue;

        // Keep only activation-scale multiplies (one constant input with expert dim, one activation input)
        for (size_t i = 0; i < 2; ++i) {
            auto const_src = get_weight_source(mul->input_value(i));
            if (auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(const_src.get_node_shared_ptr())) {
                if (!c->get_shape().empty() && c->get_shape()[0] == nodes.num_experts) {
                    auto other_src = get_weight_source(mul->input_value(1 - i));
                    if (!std::dynamic_pointer_cast<ov::op::v0::Constant>(other_src.get_node_shared_ptr())) {
                        nodes.multiplies.push_back(mul);
                        break;
                    }
                }
            }
        }
    }

    return nodes;
}

// ============================================================================
// Node transformation
// ============================================================================

void transform_tiles(LayerNodes& nodes, int64_t k_value) {
    for (auto& tile : nodes.tiles) {
        auto repeats_const =
            std::dynamic_pointer_cast<ov::op::v0::Constant>(tile->input_value(1).get_node_shared_ptr());
        auto repeats_data = repeats_const->cast_vector<int64_t>();
        repeats_data[0] = k_value;

        auto new_repeats =
            ov::op::v0::Constant::create(repeats_const->get_element_type(), repeats_const->get_shape(), repeats_data);

        tile->input(1).replace_source_output(new_repeats);
        ov::copy_runtime_info(repeats_const, new_repeats);
    }
}

void transform_constant_reshapes(LayerNodes& nodes, int64_t k_value) {
    for (auto& reshape : nodes.constant_reshapes) {
        auto shape_const =
            std::dynamic_pointer_cast<ov::op::v0::Constant>(reshape->input_value(1).get_node_shared_ptr());
        auto shape_data = shape_const->cast_vector<int64_t>();

        // Replace dim 0 (expert dimension) with K
        shape_data[0] = k_value;

        auto new_shape =
            ov::op::v0::Constant::create(shape_const->get_element_type(), shape_const->get_shape(), shape_data);

        reshape->input(1).replace_source_output(new_shape);
        ov::copy_runtime_info(shape_const, new_shape);
    }
}

void transform_dynamic_reshapes(LayerNodes& nodes) {
    // Replace dynamic reshapes with Unsqueeze at dimension 1
    for (auto& reshape : nodes.dynamic_reshapes) {
        auto data_input = reshape->input_value(0);
        auto unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(data_input, unsqueeze_axis);
        unsqueeze->set_friendly_name(reshape->get_friendly_name() + "/unsqueeze_dim1");

        ov::replace_node(reshape, unsqueeze);
        ov::copy_runtime_info(reshape, unsqueeze);
    }
}

void transform_matmuls(LayerNodes& nodes, const std::shared_ptr<ov::op::v1::Reshape>& topk_indices) {
    for (auto& matmul : nodes.matmuls) {
        auto weight_input = matmul->input_value(1);
        auto weight_source = get_weight_source(weight_input);
        auto weight_node = weight_source.get_node_shared_ptr();

        bool transformed = false;
        // Handle Multiply case: insert Gather before Multiply on expert-dimension inputs
        if (auto multiply = std::dynamic_pointer_cast<ov::op::v1::Multiply>(weight_node)) {
            for (size_t i = 0; i < 2; ++i) {
                auto mul_input = multiply->input_value(i);
                auto mul_source = get_weight_source(mul_input);
                auto mul_source_node = mul_source.get_node_shared_ptr();

                if (auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(mul_source_node)) {
                    auto shape = const_node->get_shape();
                    if (shape.size() >= 2 && shape[0] == nodes.num_experts) {
                        // Insert Gather on constant source
                        auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
                        auto gathered = std::make_shared<ov::op::v8::Gather>(mul_source, topk_indices, gather_axis);
                        gathered->set_friendly_name(mul_source_node->get_friendly_name() + "/gathered");

                        // Recreate Convert if present
                        auto mul_input_node = mul_input.get_node_shared_ptr();
                        if (auto convert = std::dynamic_pointer_cast<ov::op::v0::Convert>(mul_input_node)) {
                            auto new_convert =
                                std::make_shared<ov::op::v0::Convert>(gathered, convert->get_destination_type());
                            new_convert->set_friendly_name(convert->get_friendly_name() + "/regathered");
                            multiply->input(i).replace_source_output(new_convert);
                            ov::copy_runtime_info({mul_source_node, convert}, {gathered, new_convert});
                        } else {
                            multiply->input(i).replace_source_output(gathered);
                            ov::copy_runtime_info(mul_source_node, gathered);
                        }
                        transformed = true;
                    }
                }
            }
        } else {
            // Direct weight case: insert Gather on weight input
            auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
            auto gathered = std::make_shared<ov::op::v8::Gather>(weight_input, topk_indices, gather_axis);
            gathered->set_friendly_name(weight_input.get_node()->get_friendly_name() + "/gathered");

            matmul->input(1).replace_source_output(gathered);
            ov::copy_runtime_info(weight_input.get_node_shared_ptr(), gathered);
            transformed = true;
        }
        OPENVINO_ASSERT(transformed, "Failed to transform MatMul weights for node: ", matmul->get_friendly_name());
    }
}

void transform_adds(LayerNodes& nodes, const std::shared_ptr<ov::op::v1::Reshape>& topk_indices) {
    for (auto& add : nodes.adds) {
        bool transformed = false;
        for (size_t input_idx = 0; input_idx < 2; ++input_idx) {
            auto bias_input = add->input_value(input_idx);
            auto bias_source = get_weight_source(bias_input);
            auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(bias_source.get_node_shared_ptr());

            if (const_node) {
                auto shape = const_node->get_shape();
                if (nodes.num_experts > 0 && shape.size() >= 1 && shape[0] == nodes.num_experts) {
                    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
                    auto gathered = std::make_shared<ov::op::v8::Gather>(bias_input, topk_indices, gather_axis);
                    gathered->set_friendly_name(bias_input.get_node()->get_friendly_name() + "/gathered");

                    add->input(input_idx).replace_source_output(gathered);
                    ov::copy_runtime_info(bias_input.get_node_shared_ptr(), gathered);

                    transformed = true;
                    break;
                }
            }
        }
        OPENVINO_ASSERT(transformed, "Failed to transform Add biases for node: ", add->get_friendly_name());
    }
}

void transform_multiplies(LayerNodes& nodes, const std::shared_ptr<ov::op::v1::Reshape>& topk_indices) {
    for (auto& multiply : nodes.multiplies) {
        bool transformed = false;
        for (size_t input_idx = 0; input_idx < 2; ++input_idx) {
            auto const_input = multiply->input_value(input_idx);
            auto const_source = get_weight_source(const_input);
            auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(const_source.get_node_shared_ptr());

            if (const_node) {
                auto shape = const_node->get_shape();
                if (nodes.num_experts > 0 && shape.size() >= 1 && shape[0] == nodes.num_experts) {
                    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
                    auto gathered = std::make_shared<ov::op::v8::Gather>(const_input, topk_indices, gather_axis);
                    gathered->set_friendly_name(const_input.get_node()->get_friendly_name() + "/gathered");

                    multiply->input(input_idx).replace_source_output(gathered);
                    ov::copy_runtime_info(const_input.get_node_shared_ptr(), gathered);

                    transformed = true;
                    break;
                }
            }
        }
        OPENVINO_ASSERT(transformed,
                        "Failed to transform Multiply constants for node: ",
                        multiply->get_friendly_name());
    }
}

void transform_transpose(LayerNodes& nodes, const ov::Output<ov::Node>& router_scores) {
    if (nodes.transpose) {
        auto transpose_input = nodes.transpose->input_value(0);
        auto input_node = transpose_input.get_node_shared_ptr();

        nodes.transpose->input(0).replace_source_output(router_scores);
        ov::copy_runtime_info(input_node, router_scores.get_node_shared_ptr());
    }
}

// After transform_transpose, the router broadcast chain (Transpose → Reshape → Unsqueeze → Multiply)
// still has Reshape shape constants with dim-0 = num_experts.  Update them to k_value.
void transform_router_broadcast_chain(const RouterInfo& router, size_t num_experts) {
    if (!router.scatter_transpose)
        return;
    auto cur = router.scatter_transpose->shared_from_this();
    // Walk forward until we reach output_multiply (at most a few hops)
    for (int hops = 0; hops < 6 && cur && cur != router.output_multiply; ++hops) {
        if (auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(cur)) {
            auto shape_const =
                std::dynamic_pointer_cast<ov::op::v0::Constant>(reshape->input_value(1).get_node_shared_ptr());
            if (shape_const) {
                auto shape_data = shape_const->cast_vector<int64_t>();
                if (!shape_data.empty() && shape_data[0] == static_cast<int64_t>(num_experts)) {
                    shape_data[0] = router.k_value;
                    auto new_shape = ov::op::v0::Constant::create(shape_const->get_element_type(),
                                                                  shape_const->get_shape(),
                                                                  shape_data);
                    reshape->input(1).replace_source_output(new_shape);
                    ov::copy_runtime_info(shape_const, new_shape);
                    LOG_DEBUG("  Updated router broadcast Reshape shape[0] from " << num_experts << " to "
                                                                                  << router.k_value);
                }
            }
        }
        // Advance to the single consumer
        const auto targets = cur->output(0).get_target_inputs();
        if (targets.size() != 1)
            break;
        cur = targets.begin()->get_node()->shared_from_this();
    }
}

bool apply_layer_transformation(const RouterInfo& router, LayerNodes& nodes) {
    // Validate we have required nodes
    if (!nodes.has_required_nodes()) {
        if (nodes.constant_reshapes.empty() && nodes.dynamic_reshapes.empty()) {
            LOG_WARN("  Skipping layer: No Reshape nodes found");
        } else {
            LOG_WARN("  Skipping layer: No MatMul/Add nodes found");
        }
        return false;
    }

    // Create reshaped TopK indices [1, K] -> [K]
    auto new_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {router.k_value});
    auto topk_indices = std::make_shared<ov::op::v1::Reshape>(router.topk_indices_raw, new_shape, false);
    topk_indices->set_friendly_name(router.topk_node->get_friendly_name() + "/indices_reshaped");

    // Apply all transformations
    transform_tiles(nodes, router.k_value);
    transform_constant_reshapes(nodes, router.k_value);
    transform_dynamic_reshapes(nodes);
    transform_matmuls(nodes, topk_indices);
    transform_adds(nodes, topk_indices);
    transform_multiplies(nodes, topk_indices);
    transform_transpose(nodes, router.router_scores);
    transform_router_broadcast_chain(router, nodes.num_experts);

    LOG_INFO("DeviceRoutedMoE transformation successful");
    LOG_INFO("  Tiles: " << nodes.tiles.size() << ", ConstReshapes: " << nodes.constant_reshapes.size()
                         << ", DynReshapes: " << nodes.dynamic_reshapes.size() << ", MatMuls: " << nodes.matmuls.size()
                         << ", Adds: " << nodes.adds.size() << ", Multiplies: " << nodes.multiplies.size()
                         << ", K=" << router.k_value);

    return true;
}

}  // anonymous namespace

// ============================================================================
// Main transformation entry point
// ============================================================================

bool DeviceRoutedMoETransform::run_on_model(const std::shared_ptr<ov::Model>& model) {
    LOG_DEBUG("DeviceRoutedMoETransform: Starting transformation");

    bool model_changed = false;

    // Process each ScatterElementsUpdate node; validate each as a MoE router by topology
    for (const auto& node : model->get_ordered_ops()) {
        const bool is_scatter = std::dynamic_pointer_cast<ov::op::v3::ScatterElementsUpdate>(node) != nullptr ||
                                std::dynamic_pointer_cast<ov::op::v12::ScatterElementsUpdate>(node) != nullptr;
        if (!is_scatter)
            continue;

        // Step 1: Detect router by topology (name-independent, works for GPT-OSS and Qwen3)
        auto router = detect_router_by_topology(node);
        if (!router.has_value())
            continue;

        // Step 2: Collect expert nodes via backward BFS from output_multiply
        auto layer_nodes = collect_from_expert_output(router.value());

        // Step 3: Transform collected nodes (all-or-nothing)
        if (apply_layer_transformation(router.value(), layer_nodes)) {
            model_changed = true;
        }
    }

    return model_changed;
}

}  // namespace pass
}  // namespace npuw
}  // namespace ov
