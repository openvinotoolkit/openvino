// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "device_routed_moe_transform.hpp"

#include "../logging.hpp"
#include "../partitioning/patterns/moe.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/ops.hpp"

namespace ov {
namespace npuw {
namespace pass {

namespace opp = ov::pass::pattern;

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
    ov::Output<ov::Node> topk_softmax_scores;
    int64_t k_value;
    std::string layer_id;
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

// Extract layer ID from node name (e.g., "layers.0." from full name)
std::string extract_layer_id(const std::string& topk_name) {
    size_t layers_pos = topk_name.find("layers.");
    if (layers_pos == std::string::npos) {
        return "";
    }

    size_t start = layers_pos;
    size_t end = topk_name.find(".", start + 7);
    if (end == std::string::npos) {
        end = topk_name.find("/", start);
    }

    return (end != std::string::npos) ? topk_name.substr(start, end - start + 1) : "";
}

// Check if node name belongs to the specified layer
inline bool belongs_to_layer(const std::string& node_name, const std::string& layer_id) {
    return node_name.find(layer_id) != std::string::npos;
}

// Check if a reshape operation is unsqueeze-like (only inserts dimensions with size 1)
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
// Router processing
// ============================================================================

std::optional<RouterInfo> process_router_topk(const std::shared_ptr<ov::op::v11::TopK>& topk_node) {
    if (!topk_node || topk_node->get_mode() != ov::op::v11::TopK::Mode::MAX) {
        return std::nullopt;
    }

    std::string topk_name = topk_node->get_friendly_name();
    if (topk_name.find(ov::npuw::patterns::moe::MLP_ROUTER_NAME) == std::string::npos) {
        return std::nullopt;
    }

    // Validate TopK indices shape (batch dimension should be 1, indicates it is model for decoding)
    auto topk_indices_raw = topk_node->output(1);
    auto indices_shape = topk_indices_raw.get_partial_shape();
    if (indices_shape.rank().is_static() && indices_shape.rank().get_length() == 2) {
        if (indices_shape[0].is_static() && indices_shape[0].get_length() != 1) {
            LOG_WARN("  TopK indices batch dimension is not 1, skipping");
            return std::nullopt;
        }
    }

    // Extract K value
    auto k_input = topk_node->input_value(1);
    auto k_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(k_input.get_node_shared_ptr());
    if (!k_const) {
        LOG_WARN("  TopK K value is not a constant, skipping");
        return std::nullopt;
    }
    int64_t k_value = k_const->cast_vector<int64_t>()[0];

    // Extract layer ID
    std::string layer_id = extract_layer_id(topk_name);
    if (layer_id.empty()) {
        LOG_WARN("  Cannot extract layer ID from: " << topk_name);
        return std::nullopt;
    }

    // Find Softmax for router scores
    auto topk_values = topk_node->output(0);
    std::shared_ptr<ov::Node> topk_softmax = nullptr;
    for (const auto& target : topk_values.get_target_inputs()) {
        auto consumer = target.get_node()->shared_from_this();
        if (auto softmax = std::dynamic_pointer_cast<ov::op::v8::Softmax>(consumer)) {
            topk_softmax = softmax;
            break;
        }
    }

    if (!topk_softmax) {
        LOG_WARN("  No Softmax found for TopK values");
        return std::nullopt;
    }

    LOG_INFO("DeviceRoutedMoE: Processing router TopK: " << topk_name << " (K=" << k_value << ")");

    return RouterInfo{topk_node, topk_indices_raw, topk_softmax->output(0), k_value, layer_id};
}

// ============================================================================
// Node collection per layer
// ============================================================================

LayerNodes collect_layer_nodes(const std::shared_ptr<ov::Model>& model, const RouterInfo& router) {
    LayerNodes nodes;
    const std::string& layer_id = router.layer_id;
    int64_t k_value = router.k_value;

    // Single pass through all nodes to collect relevant operations
    for (const auto& n : model->get_ordered_ops()) {
        std::string node_name = n->get_friendly_name();

        // Skip nodes not belonging to this layer or not MoE expert nodes
        if (node_name.find(ov::npuw::patterns::moe::MLP_EXPERT_NAME) == std::string::npos ||
            !belongs_to_layer(node_name, layer_id)) {
            continue;
        }

        // Collect Tile nodes
        if (auto tile = std::dynamic_pointer_cast<ov::op::v0::Tile>(n)) {
            auto repeats_const =
                std::dynamic_pointer_cast<ov::op::v0::Constant>(tile->input_value(1).get_node_shared_ptr());
            if (repeats_const) {
                auto repeats_data = repeats_const->cast_vector<int64_t>();
                if (!repeats_data.empty() && repeats_data[0] > k_value) {
                    if (nodes.num_experts == 0) {
                        nodes.num_experts = static_cast<size_t>(repeats_data[0]);
                    }
                    nodes.tiles.push_back(tile);
                }
            }
            continue;
        }

        // Collect Reshape nodes
        if (auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(n)) {
            auto shape_const =
                std::dynamic_pointer_cast<ov::op::v0::Constant>(reshape->input_value(1).get_node_shared_ptr());

            if (!shape_const) {
                // Dynamic reshape - check if it's unsqueeze-like
                if (is_unsqueeze_like_reshape(reshape)) {
                    nodes.dynamic_reshapes.push_back(reshape);
                }
            } else {
                // Constant reshape - check if dim 0 is expert dimension
                auto shape_data = shape_const->cast_vector<int64_t>();
                if (nodes.num_experts > 0 && !shape_data.empty() &&
                    shape_data[0] == static_cast<int64_t>(nodes.num_experts)) {
                    nodes.constant_reshapes.push_back(reshape);
                }
            }
            continue;
        }

        // Collect MatMul nodes
        if (auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(n)) {
            auto weight_source = get_weight_source(matmul->input_value(1));
            auto weight_node = weight_source.get_node_shared_ptr();

            // Check if quantized weight comes from Multiply with expert-dimension constant
            if (auto multiply = std::dynamic_pointer_cast<ov::op::v1::Multiply>(weight_node)) {
                for (size_t i = 0; i < 2; ++i) {
                    auto mul_input = multiply->get_input_node_shared_ptr(i);
                    if (auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(mul_input)) {
                        auto shape = const_node->get_shape();
                        if (nodes.num_experts > 0 && shape.size() >= 2 && shape[0] == nodes.num_experts) {
                            nodes.matmuls.push_back(matmul);
                            break;
                        }
                    }
                }
            }
            continue;
        }

        // Collect Add nodes
        if (auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(n)) {
            // Check both inputs for expert-dimension bias constant
            for (size_t input_idx = 0; input_idx < 2; ++input_idx) {
                auto bias_source = get_weight_source(add->input_value(input_idx));
                auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(bias_source.get_node_shared_ptr());

                if (const_node) {
                    auto shape = const_node->get_shape();
                    if (nodes.num_experts > 0 && shape.size() >= 1 && shape[0] == nodes.num_experts) {
                        nodes.adds.push_back(add);
                        break;
                    }
                }
            }
            continue;
        }

        // Collect Multiply nodes, e.g. AWQ multiply (one input from constant, other input is not constant/convert, and
        // user is not MatMul)
        if (auto multiply = std::dynamic_pointer_cast<ov::op::v1::Multiply>(n)) {
            // Skip if this Multiply is used by MatMul (it's MatMul weights multiply, which has been processed by
            // transform_matmuls)
            bool used_by_matmul = false;
            for (const auto& output : multiply->outputs()) {
                for (const auto& target : output.get_target_inputs()) {
                    auto user = target.get_node()->shared_from_this();
                    if (std::dynamic_pointer_cast<ov::op::v0::MatMul>(user)) {
                        used_by_matmul = true;
                        break;
                    }
                }
                if (used_by_matmul)
                    break;
            }
            if (used_by_matmul) {
                continue;
            }

            // Check both inputs: one should be constant with expert dimension, other should not be constant/convert
            for (size_t const_idx = 0; const_idx < 2; ++const_idx) {
                size_t other_idx = 1 - const_idx;

                auto const_source = get_weight_source(multiply->input_value(const_idx));
                auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(const_source.get_node_shared_ptr());

                if (const_node) {
                    auto shape = const_node->get_shape();
                    if (nodes.num_experts > 0 && shape.size() >= 1 && shape[0] == nodes.num_experts) {
                        // Check if other input is not constant/convert
                        auto other_input = multiply->input_value(other_idx);
                        auto other_source = get_weight_source(other_input);
                        auto other_node = other_source.get_node_shared_ptr();

                        if (!std::dynamic_pointer_cast<ov::op::v0::Constant>(other_node)) {
                            nodes.multiplies.push_back(multiply);
                            break;
                        }
                    }
                }
            }
            continue;
        }

        // Collect Transpose node
        if (auto transpose = std::dynamic_pointer_cast<ov::op::v1::Transpose>(n)) {
            auto input_node = transpose->input_value(0).get_node_shared_ptr();
            if (std::dynamic_pointer_cast<ov::op::v3::ScatterElementsUpdate>(input_node) ||
                std::dynamic_pointer_cast<ov::op::v12::ScatterElementsUpdate>(input_node)) {
                nodes.transpose = transpose;
                // Don't break - continue collecting other nodes
            }
            continue;
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

void transform_transpose(LayerNodes& nodes, const ov::Output<ov::Node>& topk_softmax_scores) {
    if (nodes.transpose) {
        auto transpose_input = nodes.transpose->input_value(0);
        auto input_node = transpose_input.get_node_shared_ptr();

        nodes.transpose->input(0).replace_source_output(topk_softmax_scores);
        ov::copy_runtime_info(input_node, topk_softmax_scores.get_node_shared_ptr());
    }
}

bool apply_layer_transformation(const RouterInfo& router, LayerNodes& nodes) {
    // Validate we have required nodes
    if (!nodes.has_required_nodes()) {
        if (nodes.constant_reshapes.empty() && nodes.dynamic_reshapes.empty()) {
            LOG_WARN("  Skipping layer " << router.layer_id << ": No Reshape nodes found");
        } else {
            LOG_WARN("  Skipping layer " << router.layer_id << ": No MatMul/Add nodes found");
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
    transform_transpose(nodes, router.topk_softmax_scores);

    LOG_INFO("DeviceRoutedMoE transformation successful for " << router.layer_id);
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

    // Process each Router TopK node (one per MoE layer)
    for (const auto& node : model->get_ordered_ops()) {
        auto topk_node = std::dynamic_pointer_cast<ov::op::v11::TopK>(node);

        // Step 1: Process and validate router
        auto router = process_router_topk(topk_node);
        if (!router.has_value()) {
            continue;
        }

        // Step 2: Collect all nodes for this layer
        auto layer_nodes = collect_layer_nodes(model, router.value());

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
