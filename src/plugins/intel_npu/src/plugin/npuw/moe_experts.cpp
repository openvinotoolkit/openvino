// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "moe_experts.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "logging.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/validate.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace npuw {
namespace function {

// Helper function to extract K value from TopK node in router model
static std::optional<size_t> extract_k_from_router(const std::shared_ptr<ov::Model>& router_model) {
    if (!router_model) {
        LOG_ERROR("Router model is null, cannot extract K from TopK");
        return std::nullopt;
    }

    LOG_DEBUG("Searching for TopK node in router model: " << router_model->get_friendly_name());

    for (const auto& node : router_model->get_ordered_ops()) {
        if (auto topk = std::dynamic_pointer_cast<ov::op::v11::TopK>(node)) {
            LOG_DEBUG("Found TopK node: " << topk->get_friendly_name());

            // K is the second input to TopK
            auto k_input = topk->input_value(1);
            if (auto k_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(k_input.get_node_shared_ptr())) {
                auto k_data = k_const->cast_vector<int64_t>();
                if (!k_data.empty()) {
                    size_t k_value = static_cast<size_t>(k_data[0]);
                    LOG_INFO("Extracted K=" << k_value << " from TopK node in router model");
                    return k_value;
                }
            }
        }
    }

    LOG_DEBUG("TopK node not found in router model");
    return std::nullopt;
}

std::optional<MoEValidationResult> validate_and_setup_moe_expert(const std::shared_ptr<ov::Model>& model,
                                                                 size_t active_experts_num) {
    LOG_DEBUG("Validating MoE expert model...");
    LOG_BLOCK();

    MoEValidationResult result;

    // Step 1: Find Tile operation and extract expert configuration
    for (const auto& node : model->get_ordered_ops()) {
        if (auto tile = std::dynamic_pointer_cast<ov::op::v0::Tile>(node)) {
            auto repeats_input = tile->input_value(1);
            if (auto repeats_const =
                    std::dynamic_pointer_cast<ov::op::v0::Constant>(repeats_input.get_node_shared_ptr())) {
                auto repeats_data = repeats_const->cast_vector<int64_t>();

                if (!repeats_data.empty() && repeats_data[0] > 1) {
                    result.num_experts = static_cast<size_t>(repeats_data[0]);
                    result.tile_node = tile;

                    // Extract shape information
                    auto tile_output_shape = tile->output(0).get_shape();
                    if (tile_output_shape.empty() || tile_output_shape[0] % result.num_experts != 0) {
                        LOG_WARN("Invalid Tile output shape");
                        return std::nullopt;
                    }

                    result.input_token_count = tile->input_value(0).get_shape()[0];
                    result.expert_hidden_dim = tile->input_value(0).get_shape()[1];

                    LOG_DEBUG("Found Tile: num_experts=" << result.num_experts
                                                         << ", expert_hidden_dim=" << result.expert_hidden_dim
                                                         << ", input_token_count=" << result.input_token_count);

                    // Find the parameter index for Tile's input
                    auto tile_input_node = tile->input_value(0).get_node_shared_ptr();
                    std::shared_ptr<ov::Node> current_node = tile_input_node;

                    // Trace back through single-input operations to find the Parameter
                    while (current_node) {
                        if (auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(current_node)) {
                            // Found the parameter, get its index
                            const auto& params = model->get_parameters();
                            for (size_t idx = 0; idx < params.size(); ++idx) {
                                if (params[idx]->get_friendly_name() == param->get_friendly_name()) {
                                    result.expert_input_param_idx = idx;
                                    LOG_DEBUG("  Found expert input parameter at index " << idx << ": "
                                                                                         << param->get_friendly_name());
                                    break;
                                }
                            }
                            break;
                        }

                        // Move up the graph (single input ops like Convert, Reshape, etc.)
                        if (current_node->get_input_size() == 1) {
                            current_node = current_node->input_value(0).get_node_shared_ptr();
                        } else {
                            // Multi-input operation, cannot trace further
                            break;
                        }
                    }

                    break;
                }
            }
        }
    }

    if (!result.tile_node || result.num_experts == 0) {
        LOG_WARN("Could not find valid Tile operation");
        return std::nullopt;
    }

    // Step 2: Analyze output to detect stage and ReduceSum presence
    // Detect stage based on input_token_count
    // Decoding: token_count == 1, Prefill: token_count > 1
    if (result.input_token_count == 1) {
        result.is_decoding_stage = true;
        result.detected_active_experts = active_experts_num;
        LOG_DEBUG("Detected DECODING stage (input_token_count=1), K=" << result.detected_active_experts);
    } else if (result.input_token_count > 1) {
        result.is_decoding_stage = false;
        result.detected_active_experts = 1;
        LOG_DEBUG("Detected PREFILL stage (input_token_count=" << result.input_token_count << "), K=1");
    }

    return result;
}

std::optional<MoEDownstream> detect_and_transform_moe_downstream(const std::shared_ptr<ov::Model>& model,
                                                                 size_t active_experts_num) {
    LOG_DEBUG("Detecting MoE downstream pattern...");
    LOG_BLOCK();

    MoEDownstream result;
    result.active_experts_num = active_experts_num;

    // Pattern to match: Parameter -> Convert -> ReduceSum
    // Looking for a parameter with shape [N, 1, H, W] where N is total_experts_num

    for (const auto& param : model->get_parameters()) {
        auto param_shape = param->get_partial_shape();
        if (!param_shape.rank().is_static() || param_shape.rank().get_length() != 4) {
            continue;
        }

        auto shape = param_shape.to_shape();
        // Check if shape matches [N, 1, H, W] pattern
        if (shape[1] != 1 || shape[0] <= 1) {
            continue;
        }

        size_t potential_total_experts = shape[0];

        // Check if this parameter feeds into Convert -> ReduceSum
        bool found_pattern = false;
        for (const auto& param_output : param->outputs()) {
            for (const auto& target_input : param_output.get_target_inputs()) {
                auto consumer = target_input.get_node()->shared_from_this();

                // Check for Convert node
                if (auto convert_node = std::dynamic_pointer_cast<ov::op::v0::Convert>(consumer)) {
                    LOG_DEBUG("  Found Convert after Parameter: " << param->get_friendly_name());

                    // Check if Convert feeds into ReduceSum
                    for (const auto& convert_output : convert_node->outputs()) {
                        for (const auto& convert_target : convert_output.get_target_inputs()) {
                            if (auto reduce_sum = std::dynamic_pointer_cast<ov::op::v1::ReduceSum>(
                                    convert_target.get_node()->shared_from_this())) {
                                LOG_DEBUG("  Found ReduceSum after Convert");
                                LOG_DEBUG("  Pattern matched: Parameter -> Convert -> ReduceSum");

                                found_pattern = true;
                                result.total_experts_num = potential_total_experts;
                                result.expert_output_param_idx = 0;  // Find actual index

                                // Get parameter index
                                const auto& params = model->get_parameters();
                                for (size_t i = 0; i < params.size(); ++i) {
                                    if (params[i] == param) {
                                        result.expert_output_param_idx = i;
                                        break;
                                    }
                                }

                                LOG_DEBUG("  Total experts num: " << result.total_experts_num);
                                LOG_DEBUG("  Active experts num: " << result.active_experts_num);
                                LOG_DEBUG("  Expert output parameter index: " << result.expert_output_param_idx);
                                break;
                            }
                        }
                        if (found_pattern)
                            break;
                    }
                }
                if (found_pattern)
                    break;
            }
            if (found_pattern)
                break;
        }

        if (found_pattern) {
            // Validate that active_experts_num <= total_experts_num
            if (result.active_experts_num > result.total_experts_num) {
                LOG_WARN("Active experts (" << result.active_experts_num << ") > total experts ("
                                            << result.total_experts_num << "), invalid configuration");
                continue;
            }

            // Clone the model and modify the parameter shape
            auto modified_model = model->clone();

            // Find the corresponding parameter in the cloned model
            auto& cloned_params = modified_model->get_parameters();
            if (result.expert_output_param_idx < cloned_params.size()) {
                auto& target_param = cloned_params[result.expert_output_param_idx];
                auto new_shape = shape;
                new_shape[0] = result.active_experts_num;  // Change from total to active

                LOG_DEBUG("  Modifying parameter shape from " << shape << " to " << new_shape);
                LOG_DEBUG("  Parameter name: " << target_param->get_friendly_name());

                target_param->set_partial_shape(ov::PartialShape(new_shape));
                target_param->validate_and_infer_types();

                // Validate the entire model after parameter shape change
                modified_model->validate_nodes_and_infer_types();

                // Store the modified model in result
                result.modified_model = modified_model;

                // Save debug model to verify shape modification
                try {
                    std::string debug_path = "moe_downstream_model.xml";
                    ov::serialize(modified_model, debug_path);
                    LOG_INFO("Saved downstream model to: " << debug_path);
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to save downstream model: " << e.what());
                }

                LOG_INFO("Successfully detected and transformed MoE downstream pattern");
                LOG_INFO("  Expert output parameter: " << param->get_friendly_name());
                LOG_INFO("  Expert output parameter index: " << result.expert_output_param_idx);
                LOG_INFO("  Shape changed: [" << result.total_experts_num << ", 1, " << shape[2] << ", " << shape[3]
                                              << "] -> [" << result.active_experts_num << ", 1, " << shape[2] << ", "
                                              << shape[3] << "]");

                return result;
            }
        }
    }

    LOG_DEBUG("MoE downstream pattern not found");
    return std::nullopt;
}

std::optional<MoEDownstream> create_moe_downstream(const std::shared_ptr<ov::Model>& model,
                                                   const std::shared_ptr<ov::Model>& router_model) {
    LOG_DEBUG("Creating MoEDownstream from model: " << model->get_friendly_name());
    LOG_BLOCK();

    // Extract K from router model (required)
    if (!router_model) {
        LOG_ERROR("Router model is required to extract K from TopK node");
        return std::nullopt;
    }

    auto k_from_router = extract_k_from_router(router_model);
    if (!k_from_router.has_value()) {
        LOG_ERROR("Failed to extract K from router model for downstream");
        return std::nullopt;
    }

    size_t active_experts_num = k_from_router.value();
    LOG_INFO("Extracted K=" << active_experts_num << " from router model for downstream");

    auto downstream_info = detect_and_transform_moe_downstream(model, active_experts_num);
    if (downstream_info && downstream_info->is_valid()) {
        LOG_INFO("Successfully created MoEDownstream:");
        LOG_INFO("  - Total experts: " << downstream_info->total_experts_num);
        LOG_INFO("  - Active experts: " << downstream_info->active_experts_num);
        LOG_INFO("  - Expert output parameter index: " << downstream_info->expert_output_param_idx);
        LOG_INFO("  - Modified model: " << downstream_info->modified_model->get_friendly_name());

        return downstream_info;
    }

    LOG_WARN("Failed to create MoEDownstream - downstream pattern not found");
    return std::nullopt;
}

std::shared_ptr<ov::Model> transform_moe_experts(const std::shared_ptr<ov::Model>& original_model,
                                                 MoEValidationResult& validation_result,
                                                 size_t num_target_experts,
                                                 ExpertMode mode,
                                                 size_t prefill_chunk_size) {
    LOG_DEBUG("Transforming MoE model to " << num_target_experts << " expert(s), mode: "
                                           << (mode == ExpertMode::SINGLE_EXPERT ? "SINGLE_EXPERT" : "ACTIVE_EXPERTS")
                                           << ", prefill_chunk_size: " << prefill_chunk_size);
    LOG_BLOCK();

    auto model = original_model->clone();
    const auto num_experts = validation_result.num_experts;

    // Lambda: Find Tile node in the cloned model
    auto find_tile_node = [&]() -> std::shared_ptr<ov::op::v0::Tile> {
        for (const auto& node : model->get_ordered_ops()) {
            if (auto tile = std::dynamic_pointer_cast<ov::op::v0::Tile>(node)) {
                auto repeats_input = tile->input_value(1);
                if (auto repeats_const =
                        std::dynamic_pointer_cast<ov::op::v0::Constant>(repeats_input.get_node_shared_ptr())) {
                    auto repeats_data = repeats_const->cast_vector<int64_t>();
                    if (!repeats_data.empty() && repeats_data[0] == static_cast<int64_t>(num_experts)) {
                        return tile;
                    }
                }
            }
        }
        return nullptr;
    };

    // Lambda: Find MatMul consumer of Tile (possibly through Reshape)
    auto find_matmul_and_predecessor =
        [&](ov::Output<ov::Node>& tile_output) -> std::pair<std::shared_ptr<ov::Node>, ov::Output<ov::Node>> {
        for (const auto& target_input : tile_output.get_target_inputs()) {
            auto consumer = target_input.get_node()->shared_from_this();

            if (auto reshape_op = std::dynamic_pointer_cast<ov::op::v1::Reshape>(consumer)) {
                LOG_DEBUG("Found Reshape after Tile: " << reshape_op->get_friendly_name());
                for (const auto& reshape_target : reshape_op->output(0).get_target_inputs()) {
                    if (auto mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(
                            reshape_target.get_node()->shared_from_this())) {
                        LOG_DEBUG("Found MatMul after Reshape: " << mm->get_friendly_name());
                        return {mm, reshape_op->output(0)};
                    }
                }
            } else if (auto mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(consumer)) {
                LOG_DEBUG("Found MatMul directly after Tile: " << mm->get_friendly_name());
                return {mm, tile_output};
            }
        }
        return {nullptr, ov::Output<ov::Node>()};
    };

    // Lambda: Replace Tile with new Tile (or Reshape for single expert case)
    auto replace_tile_with_new_tile = [&](const ov::Output<ov::Node>& tile_input,
                                          ov::Output<ov::Node>& node_before_matmul,
                                          const std::string& friendly_name) {
        auto matmul_input_shape = node_before_matmul.get_shape();
        ov::Shape target_expert_shape = matmul_input_shape;
        target_expert_shape[0] = num_target_experts;

        // For prefill mode, update token dimension to use chunk size
        if (mode == ExpertMode::SINGLE_EXPERT && target_expert_shape.size() >= 2) {
            target_expert_shape[1] = prefill_chunk_size;
            LOG_DEBUG("Prefill mode: updating target shape token dimension to " << prefill_chunk_size);
        }

        LOG_DEBUG("Original MatMul input shape: " << matmul_input_shape);
        LOG_DEBUG("Target expert shape (" << num_target_experts << " experts): " << target_expert_shape);

        std::shared_ptr<ov::Node> new_node;

        if (num_target_experts == 1) {
            // For single expert (prefill), use Reshape since no repetition needed
            auto target_shape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                             ov::Shape{target_expert_shape.size()},
                                                                             target_expert_shape);
            new_node = std::make_shared<ov::op::v1::Reshape>(tile_input, target_shape_const, false);
            new_node->set_friendly_name(friendly_name + "_transformed_single_expert");
            LOG_DEBUG("Using Reshape for single expert");
        } else {
            // For multiple active experts (decoding), use Tile to replicate data, then Reshape to expand dims
            std::vector<int64_t> repeats_data(tile_input.get_shape().size(), 1);
            repeats_data[0] = num_target_experts;  // Repeat along first dimension

            auto repeats_const =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{repeats_data.size()}, repeats_data);
            auto tile_node = std::make_shared<ov::op::v0::Tile>(tile_input, repeats_const);
            tile_node->set_friendly_name(friendly_name + "_transformed_tile_" + std::to_string(num_target_experts));
            LOG_DEBUG("Using Tile with repeats=" << num_target_experts << " for multiple active experts");

            // After Tile: [4, 2880] -> need Reshape to [4, token_num, 2880]
            // target_expert_shape already has the correct 3D shape
            auto reshape_shape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                              ov::Shape{target_expert_shape.size()},
                                                                              target_expert_shape);
            new_node = std::make_shared<ov::op::v1::Reshape>(tile_node, reshape_shape_const, false);
            new_node->set_friendly_name(friendly_name + "_transformed_" + std::to_string(num_target_experts) +
                                        "_experts_reshape");
            LOG_DEBUG("Added Reshape after Tile to expand to 3D shape: " << target_expert_shape);
        }

        node_before_matmul.replace(new_node->output(0));
    };

    // Lambda: Fix output Reshape nodes and detect router parameter
    // Output path: Reshape -> Multiply -> [ReduceSum] -> Convert(optional) -> Result
    auto fix_output_reshapes = [&](MoEValidationResult& validation_result) {
        LOG_DEBUG("Fixing output Reshape nodes...");
        for (const auto& result : model->get_results()) {
            auto result_input = result->input_value(0);
            auto result_input_node = result_input.get_node_shared_ptr();

            // Skip Convert node if present (Result <- Convert)
            if (auto convert_node = std::dynamic_pointer_cast<ov::op::v0::Convert>(result_input_node)) {
                LOG_DEBUG("  Skipping Convert node before Result");
                result_input = convert_node->input_value(0);
                result_input_node = result_input.get_node_shared_ptr();
            }

            // Skip ReduceSum node if present (Result <- [Convert] <- ReduceSum)
            if (auto reduce_sum_node = std::dynamic_pointer_cast<ov::op::v1::ReduceSum>(result_input_node)) {
                LOG_DEBUG("  Skipping ReduceSum node before Result");
                result_input = reduce_sum_node->input_value(0);
                result_input_node = result_input.get_node_shared_ptr();
            }

            // Check for Multiply node (Result <- [Convert] <- [ReduceSum] <- Multiply)
            std::shared_ptr<ov::Node> reshape_node_ptr;
            if (auto multiply_node = std::dynamic_pointer_cast<ov::op::v1::Multiply>(result_input_node)) {
                LOG_DEBUG("  Found Multiply node before Result");

                // Detect router parameter from Multiply inputs
                if (!validation_result.router_scores_idx.has_value()) {
                    for (size_t i = 0; i < 2; ++i) {
                        auto multiply_input = multiply_node->input_value(i).get_node_shared_ptr();

                        // Skip the Reshape input, look for the other one (router parameter)
                        if (std::dynamic_pointer_cast<ov::op::v1::Reshape>(multiply_input)) {
                            continue;
                        }

                        // Trace back to find Parameter
                        std::shared_ptr<ov::Node> current_node = multiply_input;
                        while (current_node) {
                            if (auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(current_node)) {
                                // Found the parameter, get its index in original model
                                const auto& params = original_model->get_parameters();
                                for (size_t idx = 0; idx < params.size(); ++idx) {
                                    if (params[idx]->get_friendly_name() == param->get_friendly_name()) {
                                        validation_result.router_scores_idx = idx;
                                        LOG_DEBUG("  Found router scores parameter at index "
                                                  << idx << ": " << param->get_friendly_name());
                                        break;
                                    }
                                }
                                break;
                            }

                            // Move up the graph (single input ops like Convert, Reshape, etc.)
                            if (current_node->get_input_size() == 1) {
                                current_node = current_node->input_value(0).get_node_shared_ptr();
                            } else {
                                break;
                            }
                        }

                        if (validation_result.router_scores_idx.has_value()) {
                            break;
                        }
                    }
                }

                // Multiply has two inputs, one should be from Reshape
                auto multiply_input0 = multiply_node->input_value(0).get_node_shared_ptr();
                auto multiply_input1 = multiply_node->input_value(1).get_node_shared_ptr();

                if (auto reshape0 = std::dynamic_pointer_cast<ov::op::v1::Reshape>(multiply_input0)) {
                    reshape_node_ptr = reshape0;
                } else if (auto reshape1 = std::dynamic_pointer_cast<ov::op::v1::Reshape>(multiply_input1)) {
                    reshape_node_ptr = reshape1;
                }
            } else if (auto reshape_direct = std::dynamic_pointer_cast<ov::op::v1::Reshape>(result_input_node)) {
                // Direct Reshape -> Result (fallback case)
                reshape_node_ptr = reshape_direct;
            }

            if (reshape_node_ptr) {
                auto reshape_node = std::dynamic_pointer_cast<ov::op::v1::Reshape>(reshape_node_ptr);
                auto shape_input = reshape_node->input_value(1);
                if (auto shape_const =
                        std::dynamic_pointer_cast<ov::op::v0::Constant>(shape_input.get_node_shared_ptr())) {
                    auto shape_data = shape_const->cast_vector<int64_t>();

                    if (!shape_data.empty() && shape_data[0] == static_cast<int64_t>(num_experts)) {
                        LOG_DEBUG("  Found output Reshape '" << reshape_node->get_friendly_name()
                                                             << "' with shape[0]=" << shape_data[0] << ", changing to "
                                                             << num_target_experts);
                        shape_data[0] = num_target_experts;

                        auto new_shape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                                      ov::Shape{shape_data.size()},
                                                                                      shape_data);
                        auto new_output_reshape =
                            std::make_shared<ov::op::v1::Reshape>(reshape_node->input_value(0), new_shape_const, false);
                        new_output_reshape->set_friendly_name(reshape_node->get_friendly_name() + "_" +
                                                              std::to_string(num_target_experts) + "_experts");

                        reshape_node->output(0).replace(new_output_reshape->output(0));
                    }
                }
            }
        }
    };

    // Lambda: Ensure all tensors have names
    auto ensure_tensor_names = [&]() {
        LOG_DEBUG("Ensuring all tensors have names...");
        size_t in_tensor_idx = 0;
        for (auto& input : model->inputs()) {
            if (input.get_tensor().get_names().empty()) {
                std::string name = "moe_in_tensor_" + std::to_string(in_tensor_idx);
                input.get_tensor().set_names({name});
                LOG_DEBUG("  Added input tensor name: " << name);
            }
            in_tensor_idx++;
        }

        size_t out_tensor_idx = 0;
        for (auto& output : model->outputs()) {
            if (output.get_tensor().get_names().empty()) {
                std::string name = "moe_out_tensor_" + std::to_string(out_tensor_idx);
                output.get_tensor().set_names({name});
                LOG_DEBUG("  Added output tensor name: " << name);
            }
            out_tensor_idx++;
        }
    };

    // Lambda: Fix parameters with num_experts in their shape (manually, without model->reshape())
    auto fix_parameters_with_num_experts = [&]() {
        LOG_DEBUG("Fixing Parameter nodes with num_experts in shape to " << num_target_experts << " experts...");

        for (const auto& param : model->get_parameters()) {
            auto param_shape = param->get_partial_shape();
            if (param_shape.rank().is_static() && param_shape.rank().get_length() > 0) {
                auto shape = param_shape.to_shape();
                bool needs_fix = false;
                size_t fix_dim = 0;

                for (size_t i = 0; i < shape.size(); ++i) {
                    if (shape[i] == num_experts) {
                        needs_fix = true;
                        fix_dim = i;
                        break;
                    }
                }

                if (needs_fix) {
                    LOG_DEBUG("  Found Parameter '" << param->get_friendly_name() << "' with shape[" << fix_dim
                                                    << "]=" << num_experts);

                    shape[fix_dim] = num_target_experts;  // Use num_target_experts instead of hardcoded 1

                    // Directly set the new shape on the parameter
                    param->set_partial_shape(ov::PartialShape(shape));
                    param->validate_and_infer_types();
                    LOG_DEBUG("    Successfully updated parameter shape to " << num_target_experts);
                }
            }
        }
    };

    // Lambda: Fix token count for prefill mode - trace from Tile to Parameter and update
    auto fix_token_count_for_prefill = [&]() {
        if (mode != ExpertMode::SINGLE_EXPERT) {
            return;  // Only apply to prefill mode
        }

        LOG_DEBUG("Fixing token count from " << validation_result.input_token_count << " to " << prefill_chunk_size
                                             << " for prefill mode...");

        // Find the Tile node
        auto tile_node = find_tile_node();
        if (!tile_node) {
            LOG_WARN("Cannot find Tile node for token count fixing");
            return;
        }

        // Trace back from Tile input to find Parameters
        std::set<std::shared_ptr<ov::op::v0::Parameter>> params_to_fix;
        std::function<void(const ov::Output<ov::Node>&)> trace_to_params;
        trace_to_params = [&](const ov::Output<ov::Node>& output) {
            auto node = output.get_node_shared_ptr();

            if (auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(node)) {
                params_to_fix.insert(param);
                return;
            }

            // Skip Convert nodes during tracing
            if (auto convert = std::dynamic_pointer_cast<ov::op::v0::Convert>(node)) {
                LOG_DEBUG("  Skipping Convert node during trace: " << convert->get_friendly_name());
            }

            // Recursively trace inputs
            for (size_t i = 0; i < node->get_input_size(); ++i) {
                trace_to_params(node->input_value(i));
            }
        };

        trace_to_params(tile_node->input_value(0));

        // Also find router parameter from Result -> [Convert] -> Multiply
        LOG_DEBUG("Finding router parameter from output path...");
        for (const auto& result_node : model->get_results()) {
            auto current = result_node->input_value(0).get_node_shared_ptr();

            // Skip Convert if present
            if (auto convert = std::dynamic_pointer_cast<ov::op::v0::Convert>(current)) {
                LOG_DEBUG("  Skipping Convert in output path");
                current = convert->input_value(0).get_node_shared_ptr();
            }

            // Skip ReduceSum if present
            if (auto reduce_sum = std::dynamic_pointer_cast<ov::op::v1::ReduceSum>(current)) {
                LOG_DEBUG("  Skipping ReduceSum in output path");
                current = reduce_sum->input_value(0).get_node_shared_ptr();
            }

            // Find Multiply node
            if (auto multiply = std::dynamic_pointer_cast<ov::op::v1::Multiply>(current)) {
                LOG_DEBUG("  Found Multiply node in output path: " << multiply->get_friendly_name());

                // Check both inputs, one should be the router parameter
                for (size_t i = 0; i < 2; ++i) {
                    auto multiply_input = multiply->input_value(i).get_node_shared_ptr();

                    // Skip Reshape input (that's the expert output)
                    if (std::dynamic_pointer_cast<ov::op::v1::Reshape>(multiply_input)) {
                        continue;
                    }

                    // Trace to parameter
                    trace_to_params(multiply->input_value(i));
                }
            }
        }

        // Fix Parameter shapes
        size_t original_token_count = validation_result.input_token_count;
        for (const auto& param : params_to_fix) {
            auto param_shape = param->get_partial_shape();
            if (param_shape.rank().is_static() && param_shape.rank().get_length() >= 2) {
                auto shape = param_shape.to_shape();

                // Find dimension with original token count
                for (size_t i = 0; i < shape.size(); ++i) {
                    if (shape[i] == original_token_count) {
                        LOG_DEBUG("  Updating Parameter '" << param->get_friendly_name() << "' shape[" << i << "] from "
                                                           << original_token_count << " to " << prefill_chunk_size);
                        shape[i] = prefill_chunk_size;
                    }
                }

                param->set_partial_shape(ov::PartialShape(shape));
                param->validate_and_infer_types();
            }
        }

        // Trigger shape inference to propagate changes through the model
        LOG_DEBUG("Triggering shape inference after token count changes...");
        model->validate_nodes_and_infer_types();
    };

    // Lambda: Save model for debugging
    auto save_debug_model = [&]() {
        try {
            std::string debug_path = "moe_transformed_" + std::to_string(num_target_experts) + "_experts_model.xml";
            ov::serialize(model, debug_path);
            LOG_INFO("Saved transformed expert model to: " << debug_path);
        } catch (const std::exception& e) {
            LOG_WARN("Failed to save transformed expert model for debugging: " << e.what());
        }
    };

    // Main transformation flow
    auto tile_op = find_tile_node();
    if (!tile_op) {
        LOG_ERROR("Could not find Tile operation in cloned model");
        return nullptr;
    }

    auto tile_input = tile_op->input_value(0);
    auto tile_output = tile_op->output(0);

    auto [matmul_node, node_before_matmul] = find_matmul_and_predecessor(tile_output);
    if (!matmul_node) {
        LOG_ERROR("Could not find MatMul operation after Tile");
        return nullptr;
    }

    fix_token_count_for_prefill();  // Fix token count BEFORE replacing Tile (Tile will be replaced)
    replace_tile_with_new_tile(tile_input, node_before_matmul, tile_op->get_friendly_name());
    fix_output_reshapes(validation_result);
    fix_parameters_with_num_experts();
    ensure_tensor_names();

    model->validate_nodes_and_infer_types();
    LOG_DEBUG("Successfully transformed to " << num_target_experts << " expert(s) model");
    std::cout << "Successfully transformed to " << num_target_experts << " expert(s) model." << std::endl;

    save_debug_model();

    return model;
}

std::optional<MoEExperts> MoEExperts::from(const std::shared_ptr<ov::Model>& model,
                                           const std::shared_ptr<ov::Model>& router_model,
                                           size_t prefill_chunk_size) {
    LOG_DEBUG("Creating MoEExperts from model: " << model->get_friendly_name());
    LOG_BLOCK();

    // Step 0: Extract actual K from router model (required)
    if (!router_model) {
        LOG_ERROR("Router model is required to extract K from TopK node");
        return std::nullopt;
    }

    auto k_from_router = extract_k_from_router(router_model);
    if (!k_from_router.has_value()) {
        LOG_ERROR("Failed to extract K from router model");
        return std::nullopt;
    }

    size_t actual_active_experts_num = k_from_router.value();
    LOG_INFO("Extracted K=" << actual_active_experts_num << " from router model");

    // Step 1: Validate the model and extract expert information (including stage detection)
    auto validation_result = validate_and_setup_moe_expert(model, actual_active_experts_num);
    if (!validation_result || !validation_result->is_valid()) {
        LOG_WARN("Model validation failed for MoE expert pattern");
        return std::nullopt;
    }

    // Step 2: Use auto-detected parameters
    const bool is_decoding = validation_result->is_decoding_stage;
    const size_t num_target_experts = validation_result->detected_active_experts;
    const ExpertMode mode = is_decoding ? ExpertMode::ACTIVE_EXPERTS : ExpertMode::SINGLE_EXPERT;

    LOG_INFO("Auto-detected MoE configuration:");
    LOG_INFO("  Stage: " << (is_decoding ? "DECODING" : "PREFILL"));
    LOG_INFO("  Mode: " << (mode == ExpertMode::SINGLE_EXPERT ? "SINGLE_EXPERT" : "ACTIVE_EXPERTS"));
    LOG_INFO("  Target num experts: " << num_target_experts);

    // Step 3: Transform the model to target number of experts using configured chunk size
    auto transformed_model =
        transform_moe_experts(model, *validation_result, num_target_experts, mode, prefill_chunk_size);
    if (!transformed_model) {
        LOG_WARN("Failed to transform model to " << num_target_experts << " expert(s)");
        return std::nullopt;
    }

    // Step 5: Populate MoEExperts structure
    MoEExperts moe_experts;
    moe_experts._num_experts = validation_result->num_experts;
    moe_experts._expert_hidden_dim = validation_result->expert_hidden_dim;
    moe_experts._input_token_count = validation_result->input_token_count;  // Keep original token count
    moe_experts._chunk_token_count = is_decoding ? 0 : prefill_chunk_size;  // Set chunk size for prefill
    moe_experts._mode = mode;
    // Store the actual active expert number (K from router)
    // For prefill: model is transformed to 1 expert, but _num_active_experts stores actual K
    // For decoding: model is transformed to K experts, _num_active_experts also stores K
    moe_experts._num_active_experts = actual_active_experts_num;
    moe_experts._transformed_model = transformed_model;
    moe_experts._router_scores_idx = validation_result->router_scores_idx;
    moe_experts._expert_input_param_idx = validation_result->expert_input_param_idx;

    // Step 6: Extract input/output information
    if (!moe_experts.is_valid()) {
        LOG_WARN("Created MoEExperts structure is invalid");
        return std::nullopt;
    }

    LOG_INFO("Successfully created MoEExperts:");
    LOG_INFO("  - Mode: " << (mode == ExpertMode::SINGLE_EXPERT ? "SINGLE_EXPERT" : "ACTIVE_EXPERTS"));
    LOG_INFO("  - Total number of experts: " << moe_experts._num_experts);
    LOG_INFO("  - Num active experts: " << moe_experts._num_active_experts);
    LOG_INFO("  - Expert hidden dim: " << moe_experts._expert_hidden_dim);
    LOG_INFO("  - Transformed model: " << transformed_model->get_friendly_name());

    return moe_experts;
}

}  // namespace function

namespace compiled {

MoEExperts::MoEExperts(const function::MoEExperts& func_moe) {
    num_experts = func_moe._num_experts;
    expert_hidden_dim = func_moe._expert_hidden_dim;
    num_active_experts = func_moe._num_active_experts;
    input_token_count = func_moe._input_token_count;
    chunk_token_count = func_moe._chunk_token_count;
    mode = func_moe._mode;
    _model_to_compile = func_moe._transformed_model;
    _router_scores_idx = func_moe._router_scores_idx;
    _expert_input_param_idx = func_moe._expert_input_param_idx;

    LOG_DEBUG("Created compiled::MoEExperts:");
    LOG_DEBUG("  Mode: " << (mode == function::ExpertMode::SINGLE_EXPERT ? "SINGLE_EXPERT" : "ACTIVE_EXPERTS"));
    LOG_DEBUG("  Total experts: " << num_experts);
    LOG_DEBUG("  Active experts: " << num_active_experts);
    LOG_DEBUG("  Input token count: " << input_token_count);
    LOG_DEBUG("  Chunk token count: " << chunk_token_count);
    if (_router_scores_idx.has_value()) {
        LOG_DEBUG("  Router scores parameter index: " << _router_scores_idx.value());
    }
    if (_expert_input_param_idx.has_value()) {
        LOG_DEBUG("  Expert input parameter index: " << _expert_input_param_idx.value());
    }
}

void MoEExperts::set_compiled_model(ov::SoPtr<ov::ICompiledModel>&& compiled_model) {
    _compiled_model = std::move(compiled_model);
    _model_to_compile.reset();  // Free memory after compilation

    LOG_DEBUG("Set compiled model for MoE experts");
}

MoEDownstream::MoEDownstream(const function::MoEDownstream& func_downstream) {
    total_experts_num = func_downstream.total_experts_num;
    active_experts_num = func_downstream.active_experts_num;
    expert_output_param_idx = func_downstream.expert_output_param_idx;
    _model_to_compile = func_downstream.modified_model;

    LOG_DEBUG("Created compiled::MoEDownstream with " << total_experts_num << " total experts and "
                                                      << active_experts_num << " active experts");
}

void MoEDownstream::set_compiled_model(ov::SoPtr<ov::ICompiledModel>&& compiled_model) {
    _compiled_model = std::move(compiled_model);
    _model_to_compile.reset();  // Free memory after compilation

    LOG_DEBUG("Set compiled model for MoE downstream");
}

}  // namespace compiled

namespace runtime {
namespace moe_experts {

// TODO: Implement runtime execution logic

}  // namespace moe_experts
}  // namespace runtime

}  // namespace npuw
}  // namespace ov
