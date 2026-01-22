// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "device_routed_moe_transform.hpp"

#include "../logging.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/ops.hpp"

namespace ov {
namespace npuw {
namespace pass {

namespace opp = ov::pass::pattern;

/**
 * @brief DeviceRoutedMoETransform Implementation
 *
 * This pass transforms batched MoE expert computations to use Gather-based
 * dynamic weight selection driven by Router's TopK indices.
 *
 * TWO-PHASE APPROACH:
 * Phase 1: Collect all nodes that need transformation for each layer
 * Phase 2: Only transform if ALL required nodes are found (all-or-nothing)
 *
 * Key Steps:
 * 1. Find Router TopK nodes in the model
 * 2. Extract TopK indices and Softmax scores
 * 3. Collect all Expert Tile, Reshape, MatMul, Add nodes
 * 4. If all required nodes found, insert Gather(weights/bias, topk_indices, axis=0)
 * 5. Replace router scores with TopK Softmax outputs
 */

bool DeviceRoutedMoETransform::run_on_model(const std::shared_ptr<ov::Model>& model) {
    LOG_DEBUG("DeviceRoutedMoETransform: Starting transformation");

    bool model_changed = false;

    // Helper: Trace back from MatMul/Add input to find the actual weight/bias data source
    auto get_weight_source = [](const ov::Output<ov::Node>& input) -> ov::Output<ov::Node> {
        auto node = input.get_node_shared_ptr();
        if (auto convert = std::dynamic_pointer_cast<ov::op::v0::Convert>(node)) {
            return convert->input_value(0);
        }
        return input;
    };

    // =========================================================================
    // Find all Router TopK nodes and process each layer
    // =========================================================================
    for (const auto& node : model->get_ordered_ops()) {
        auto topk_node = std::dynamic_pointer_cast<ov::op::v11::TopK>(node);
        if (!topk_node || topk_node->get_mode() != ov::op::v11::TopK::Mode::MAX) {
            continue;
        }

        std::string topk_name = topk_node->get_friendly_name();
        bool is_router = (topk_name.find("router") != std::string::npos ||
                          topk_name.find("gate") != std::string::npos || topk_name.find("expert") != std::string::npos);
        if (!is_router) {
            continue;
        }

        // Extract TopK indices and values
        auto topk_indices_raw = topk_node->output(1);
        auto topk_values = topk_node->output(0);
        LOG_INFO("DeviceRoutedMoE: Processing router TopK: " << topk_name);

        auto indices_shape = topk_indices_raw.get_partial_shape();
        if (indices_shape.rank().is_static() && indices_shape.rank().get_length() == 2) {
            if (indices_shape[0].is_static() && indices_shape[0].get_length() != 1) {
                LOG_WARN("  TopK indices batch dimension is not 1, skipping");
                continue;
            }
        }

        auto k_input = topk_node->input_value(1);
        auto k_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(k_input.get_node_shared_ptr());
        if (!k_const) {
            LOG_WARN("  TopK K value is not a constant, skipping");
            continue;
        }
        int64_t k_value = k_const->cast_vector<int64_t>()[0];

        // Extract layer ID with trailing delimiter
        std::string layer_id = "";
        size_t layers_pos = topk_name.find("layers.");
        if (layers_pos != std::string::npos) {
            size_t start = layers_pos;
            size_t end = topk_name.find(".", start + 7);
            if (end == std::string::npos) {
                end = topk_name.find("/", start);
            }
            if (end != std::string::npos) {
                layer_id = topk_name.substr(start, end - start + 1);
            }
        }

        if (layer_id.empty()) {
            LOG_WARN("  Cannot extract layer ID, skipping");
            continue;
        }

        // Find Softmax for router scores
        std::shared_ptr<ov::Node> topk_softmax = nullptr;
        for (const auto& target : topk_values.get_target_inputs()) {
            auto consumer = target.get_node()->shared_from_this();
            if (auto softmax = std::dynamic_pointer_cast<ov::op::v8::Softmax>(consumer)) {
                topk_softmax = softmax;
                break;
            }
        }
        auto topk_softmax_scores = topk_softmax->output(0);

        // =====================================================================
        // PHASE 1: COLLECT all nodes for this layer
        // =====================================================================
        std::vector<std::shared_ptr<ov::op::v0::Tile>> tiles_to_transform;
        std::vector<std::shared_ptr<ov::op::v1::Reshape>> reshapes_to_transform;
        std::vector<std::shared_ptr<ov::op::v1::Reshape>> concat_reshapes_to_transform;
        std::vector<std::shared_ptr<ov::op::v0::MatMul>> matmuls_to_transform;
        std::vector<std::shared_ptr<ov::op::v1::Add>> adds_to_transform;
        std::shared_ptr<ov::op::v1::Transpose> transpose_to_transform = nullptr;
        size_t num_experts = 0;

        // Collect Tiles
        for (const auto& n : model->get_ordered_ops()) {
            auto tile = std::dynamic_pointer_cast<ov::op::v0::Tile>(n);
            if (!tile)
                continue;

            std::string node_name = tile->get_friendly_name();
            if (node_name.find("expert") == std::string::npos)
                continue;
            if (node_name.find(layer_id) == std::string::npos)
                continue;

            auto repeats_input = tile->input_value(1);
            auto repeats_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(repeats_input.get_node_shared_ptr());
            if (!repeats_const)
                continue;

            auto repeats_data = repeats_const->cast_vector<int64_t>();
            if (repeats_data.empty() || repeats_data[0] <= k_value)
                continue;

            if (num_experts == 0) {
                num_experts = static_cast<size_t>(repeats_data[0]);
            }

            tiles_to_transform.push_back(tile);
        }

        // Collect Reshapes (only Constant shapes)
        for (const auto& n : model->get_ordered_ops()) {
            auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(n);
            if (!reshape)
                continue;

            std::string node_name = reshape->get_friendly_name();
            if (node_name.find("expert") == std::string::npos)
                continue;
            if (node_name.find("mlp") == std::string::npos)
                continue;
            if (node_name.find(layer_id) == std::string::npos)
                continue;

            auto shape_input = reshape->input_value(1);
            auto shape_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(shape_input.get_node_shared_ptr());
            if (!shape_const) {
                // TODO: TEMPORARY handling for non-Constant shape (likely Concat)
                // This is a simplified approach that collects all non-Constant reshapes
                // More rigorous validation is needed:
                //   - Verify the reshape is actually Unsqueeze-like (inserting dimension)
                //   - Check input/output shape compatibility
                //   - Ensure first dimension is num_experts
                //   - Validate which dimension is being inserted (should be dim 1)
                concat_reshapes_to_transform.push_back(reshape);
                continue;
            }

            // Constant shape case
            auto shape_data = shape_const->cast_vector<int64_t>();
            bool has_expert_dim = false;
            for (const auto& dim : shape_data) {
                if (num_experts > 0 && dim == static_cast<int64_t>(num_experts)) {
                    has_expert_dim = true;
                    break;
                }
            }

            if (has_expert_dim) {
                reshapes_to_transform.push_back(reshape);
            }
        }

        // Collect MatMuls
        for (const auto& n : model->get_ordered_ops()) {
            auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(n);
            if (!matmul)
                continue;

            std::string node_name = matmul->get_friendly_name();
            if (node_name.find("expert") == std::string::npos)
                continue;
            if (node_name.find(layer_id) == std::string::npos)
                continue;

            auto weight_input = matmul->input_value(1);
            auto weight_source = get_weight_source(weight_input);
            auto weight_node = weight_source.get_node_shared_ptr();

            if (auto multiply = std::dynamic_pointer_cast<ov::op::v1::Multiply>(weight_node)) {
                for (size_t i = 0; i < 2; ++i) {
                    auto mul_input = multiply->get_input_node_shared_ptr(i);
                    if (auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(mul_input)) {
                        auto shape = const_node->get_shape();
                        if (num_experts > 0 && shape.size() >= 2 && shape[0] == num_experts) {
                            matmuls_to_transform.push_back(matmul);
                            break;
                        }
                    }
                }
            }
        }

        // Collect Adds
        for (const auto& n : model->get_ordered_ops()) {
            auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(n);
            if (!add)
                continue;

            std::string node_name = add->get_friendly_name();
            if (node_name.find("expert") == std::string::npos)
                continue;
            if (node_name.find(layer_id) == std::string::npos)
                continue;

            bool has_expert_bias = false;
            for (size_t input_idx = 0; input_idx < 2; ++input_idx) {
                auto bias_input = add->input_value(input_idx);
                auto bias_source = get_weight_source(bias_input);
                auto bias_node = bias_source.get_node_shared_ptr();

                if (auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(bias_node)) {
                    auto shape = const_node->get_shape();
                    if (num_experts > 0 && shape.size() >= 1 && shape[0] == num_experts) {
                        has_expert_bias = true;
                        break;
                    }
                }
            }

            if (has_expert_bias) {
                adds_to_transform.push_back(add);
            }
        }

        // Collect Transpose
        for (const auto& n : model->get_ordered_ops()) {
            auto transpose = std::dynamic_pointer_cast<ov::op::v1::Transpose>(n);
            if (!transpose)
                continue;

            std::string node_name = transpose->get_friendly_name();
            if (node_name.find(layer_id) == std::string::npos)
                continue;
            if (node_name.find("router") == std::string::npos && node_name.find("experts") == std::string::npos)
                continue;

            auto transpose_input = transpose->input_value(0);
            auto input_node = transpose_input.get_node_shared_ptr();

            if (std::dynamic_pointer_cast<ov::op::v3::ScatterElementsUpdate>(input_node) ||
                std::dynamic_pointer_cast<ov::op::v12::ScatterElementsUpdate>(input_node)) {
                transpose_to_transform = transpose;
                break;
            }
        }

        // Check collection results
        if (reshapes_to_transform.empty() && concat_reshapes_to_transform.empty()) {
            LOG_WARN("  Skipping layer " << layer_id << ": No Reshape nodes");
            continue;
        }

        if (matmuls_to_transform.empty() && adds_to_transform.empty()) {
            LOG_WARN("  Skipping layer " << layer_id << ": No MatMul/Add nodes");
            continue;
        }

        // =====================================================================
        // PHASE 2: TRANSFORM all collected nodes
        // =====================================================================
        // Create reshaped TopK indices [1, K] -> [K] (only now after validation passed)
        auto new_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {k_value});
        auto topk_indices = std::make_shared<ov::op::v1::Reshape>(topk_indices_raw, new_shape, false);
        topk_indices->set_friendly_name(topk_name + "/indices_reshaped");

        // Transform Tiles
        for (auto& tile : tiles_to_transform) {
            auto repeats_input = tile->input_value(1);
            auto repeats_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(repeats_input.get_node_shared_ptr());
            auto repeats_data = repeats_const->cast_vector<int64_t>();
            repeats_data[0] = k_value;

            auto new_repeats = ov::op::v0::Constant::create(repeats_const->get_element_type(),
                                                            repeats_const->get_shape(),
                                                            repeats_data);

            tile->input(1).replace_source_output(new_repeats);
            ov::copy_runtime_info(repeats_const, new_repeats);
        }

        // Transform Reshapes (Constant-based)
        for (auto& reshape : reshapes_to_transform) {
            auto shape_input = reshape->input_value(1);
            auto shape_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(shape_input.get_node_shared_ptr());
            auto shape_data = shape_const->cast_vector<int64_t>();

            for (size_t i = 0; i < shape_data.size(); ++i) {
                if (shape_data[i] == static_cast<int64_t>(num_experts)) {
                    shape_data[i] = k_value;
                    break;
                }
            }

            auto new_shape =
                ov::op::v0::Constant::create(shape_const->get_element_type(), shape_const->get_shape(), shape_data);

            reshape->input(1).replace_source_output(new_shape);
            ov::copy_runtime_info(shape_const, new_shape);
        }

        // Transform Reshapes (Concat-based) - replace with Unsqueeze on dim 1
        // TODO: TEMPORARY transformation - assumes all Concat-based reshapes insert at dim 1
        // This is a simplified approach that may not work for all cases.
        // More rigorous transformation needed:
        //   - Analyze actual Concat structure to determine inserted dimension
        //   - Verify shape compatibility before/after transformation
        //   - Handle edge cases (multiple insertions, dynamic shapes, etc.)
        for (auto& reshape : concat_reshapes_to_transform) {
            auto data_input = reshape->input_value(0);

            // Insert unsqueeze at dimension 1 (assumed location)
            auto unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
            auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(data_input, unsqueeze_axis);
            unsqueeze->set_friendly_name(reshape->get_friendly_name() + "/unsqueeze_dim1");

            ov::replace_node(reshape, unsqueeze);
            ov::copy_runtime_info(reshape, unsqueeze);
        }

        // Transform MatMuls
        for (auto& matmul : matmuls_to_transform) {
            auto weight_input = matmul->input_value(1);
            auto weight_source = get_weight_source(weight_input);
            auto weight_node = weight_source.get_node_shared_ptr();

            // If the weight is from a Multiply, insert Gather on each Multiply input separately
            if (auto multiply = std::dynamic_pointer_cast<ov::op::v1::Multiply>(weight_node)) {
                // Gather on both Multiply inputs (before Multiply operation)
                for (size_t i = 0; i < 2; ++i) {
                    auto mul_input = multiply->input_value(i);

                    // Trace back through Convert to find the actual source
                    auto mul_source = get_weight_source(mul_input);
                    auto mul_source_node = mul_source.get_node_shared_ptr();

                    // Check if this input has expert dimension
                    if (auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(mul_source_node)) {
                        auto shape = const_node->get_shape();
                        if (shape.size() >= 2 && shape[0] == num_experts) {
                            // Insert Gather BEFORE Convert (on the Constant source)
                            auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
                            auto gathered = std::make_shared<ov::op::v8::Gather>(mul_source, topk_indices, gather_axis);
                            gathered->set_friendly_name(mul_source_node->get_friendly_name() + "/gathered");

                            // If there was a Convert, recreate it after Gather
                            auto mul_input_node = mul_input.get_node_shared_ptr();
                            if (auto convert = std::dynamic_pointer_cast<ov::op::v0::Convert>(mul_input_node)) {
                                auto new_convert =
                                    std::make_shared<ov::op::v0::Convert>(gathered, convert->get_destination_type());
                                new_convert->set_friendly_name(convert->get_friendly_name() + "/regathered");
                                multiply->input(i).replace_source_output(new_convert);
                                ov::copy_runtime_info({mul_source_node, convert}, {gathered, new_convert});
                            } else {
                                // No Convert, directly connect Gather to Multiply
                                multiply->input(i).replace_source_output(gathered);
                                ov::copy_runtime_info(mul_source_node, gathered);
                            }
                        }
                    }
                }
            } else {
                // Direct weight without Multiply - insert Gather as before
                auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
                auto gathered = std::make_shared<ov::op::v8::Gather>(weight_input, topk_indices, gather_axis);
                gathered->set_friendly_name(weight_input.get_node()->get_friendly_name() + "/gathered");

                matmul->input(1).replace_source_output(gathered);
                ov::copy_runtime_info(weight_input.get_node_shared_ptr(), gathered);
            }
        }

        // Transform Adds
        for (auto& add : adds_to_transform) {
            for (size_t input_idx = 0; input_idx < 2; ++input_idx) {
                auto bias_input = add->input_value(input_idx);
                auto bias_source = get_weight_source(bias_input);
                auto bias_node = bias_source.get_node_shared_ptr();

                if (auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(bias_node)) {
                    auto shape = const_node->get_shape();
                    if (num_experts > 0 && shape.size() >= 1 && shape[0] == num_experts) {
                        auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
                        auto gathered = std::make_shared<ov::op::v8::Gather>(bias_input, topk_indices, gather_axis);
                        gathered->set_friendly_name(bias_input.get_node()->get_friendly_name() + "/gathered");

                        add->input(input_idx).replace_source_output(gathered);
                        ov::copy_runtime_info(bias_input.get_node_shared_ptr(), gathered);
                        break;
                    }
                }
            }
        }

        // Transform Transpose
        if (transpose_to_transform) {
            auto transpose_input = transpose_to_transform->input_value(0);
            auto input_node = transpose_input.get_node_shared_ptr();

            transpose_to_transform->input(0).replace_source_output(topk_softmax_scores);
            ov::copy_runtime_info(input_node, topk_softmax_scores.get_node_shared_ptr());
        }

        LOG_INFO("DeviceRoutedMoE transformation successful for " << layer_id);
        LOG_INFO("  Tiles: " << tiles_to_transform.size() << ", Reshapes: " << reshapes_to_transform.size()
                             << ", ConcatReshapes: " << concat_reshapes_to_transform.size()
                             << ", MatMuls: " << matmuls_to_transform.size() << ", Adds: " << adds_to_transform.size()
                             << ", K=" << k_value);
        model_changed = true;
    }

    return model_changed;
}

bool DeviceRoutedMoEOptimization::run_on_model(const std::shared_ptr<ov::Model>& model) {
    DeviceRoutedMoETransform transform;
    return transform.run_on_model(model);
}

}  // namespace pass
}  // namespace npuw
}  // namespace ov
