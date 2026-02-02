// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "moe_transformation.hpp"

#include "moe_unroll_patterns.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/manager.hpp"
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

// Helper function to build parameter mapping from RTInfo metadata
static std::map<size_t, std::vector<size_t>> build_parameter_mapping_from_rtinfo(
    const std::shared_ptr<ov::Model>& original_model,
    const std::shared_ptr<ov::Model>& transformed_model) {
    std::map<size_t, std::vector<size_t>> param_mapping;
    LOG_DEBUG("Building parameter mapping from RTInfo...");

    // Get original model parameters for index lookup
    const auto& original_params = original_model->get_parameters();
    const auto& transformed_params = transformed_model->get_parameters();

    // Walk through transformed model parameters and extract RTInfo
    for (size_t new_param_idx = 0; new_param_idx < transformed_params.size(); ++new_param_idx) {
        const auto& param = transformed_params[new_param_idx];
        const auto& rt_info = param->get_rt_info();

        // Check if this parameter has MoE RTInfo metadata
        if (rt_info.count("moe_original_param") > 0 && rt_info.count("moe_expert_index") > 0) {
            std::string original_param_name = rt_info.at("moe_original_param").as<std::string>();
            int64_t expert_idx = rt_info.at("moe_expert_index").as<int64_t>();

            // Find the original parameter index by name
            size_t original_param_idx = 0;
            bool found = false;
            for (size_t i = 0; i < original_params.size(); ++i) {
                if (original_params[i]->get_friendly_name() == original_param_name) {
                    original_param_idx = i;
                    found = true;
                    break;
                }
            }

            if (found) {
                // Add to mapping: original_idx -> [unrolled_indices]
                param_mapping[original_param_idx].push_back(new_param_idx);

                LOG_DEBUG("  Mapped: original param[" << original_param_idx << "] '" << original_param_name
                                                      << "' -> transformed param[" << new_param_idx << "] (expert "
                                                      << expert_idx << ")");
            } else {
                LOG_WARN("  Could not find original parameter '" << original_param_name << "' in original model");
            }
        }
    }

    LOG_INFO("Parameter mapping built: " << param_mapping.size() << " original parameters unrolled");
    for (const auto& entry : param_mapping) {
        LOG_DEBUG("  Original param[" << entry.first << "] -> " << entry.second.size() << " unrolled params");
    }

    return param_mapping;
}

// Helper: Trace back from a node to find the source Parameter
static std::optional<size_t> trace_to_parameter(const std::shared_ptr<ov::Node>& start_node,
                                                const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::Node> current_node = start_node;

    while (current_node) {
        if (auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(current_node)) {
            // Found parameter, get its index
            const auto& params = model->get_parameters();
            for (size_t idx = 0; idx < params.size(); ++idx) {
                if (params[idx]->get_friendly_name() == param->get_friendly_name()) {
                    return idx;
                }
            }
            return std::nullopt;
        }

        // Only skip Convert and Reshape nodes when tracing back
        if (current_node->get_input_size() == 1) {
            if (std::dynamic_pointer_cast<ov::op::v0::Convert>(current_node) ||
                std::dynamic_pointer_cast<ov::op::v1::Reshape>(current_node)) {
                current_node = current_node->input_value(0).get_node_shared_ptr();
                continue;
            }
        }

        // Cannot trace further through this node
        break;
    }

    return std::nullopt;
}

// Helper: Find Tile node and extract expert configuration
static bool find_tile_and_extract_config(const std::shared_ptr<ov::Model>& model, MoEStructureInfo& info) {
    for (const auto& node : model->get_ordered_ops()) {
        auto tile = std::dynamic_pointer_cast<ov::op::v0::Tile>(node);
        if (!tile)
            continue;

        auto repeats_input = tile->input_value(1);
        auto repeats_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(repeats_input.get_node_shared_ptr());
        if (!repeats_const)
            continue;

        auto repeats_data = repeats_const->cast_vector<int64_t>();
        if (repeats_data.empty() || repeats_data[0] <= 1)
            continue;

        // Found valid Tile node
        info.num_experts = static_cast<size_t>(repeats_data[0]);
        info.expert_input_tile_node = tile;

        // Validate and extract shape information
        auto tile_output_shape = tile->output(0).get_shape();
        if (tile_output_shape.empty() || tile_output_shape[0] % info.num_experts != 0) {
            LOG_WARN("Invalid Tile output shape");
            return false;
        }

        info.input_token_count = tile->input_value(0).get_shape()[0];
        info.expert_hidden_dim = tile->input_value(0).get_shape()[1];

        LOG_DEBUG("Found Tile: num_experts=" << info.num_experts << ", expert_hidden_dim=" << info.expert_hidden_dim
                                             << ", input_token_count=" << info.input_token_count);

        // Find the parameter index for Tile's input
        auto tile_input_node = tile->input_value(0).get_node_shared_ptr();
        info.expert_input_param_idx = trace_to_parameter(tile_input_node, model);

        if (info.expert_input_param_idx.has_value()) {
            LOG_DEBUG("  Found expert input parameter at index " << info.expert_input_param_idx.value());
        }

        return true;
    }

    LOG_WARN("Could not find valid Tile operation");
    return false;
}

// Helper: Find router scores parameter from output path
// Output path: Result <- [Convert] <- [ReduceSum] <- Multiply <- router_param
static bool find_router_scores_from_output(const std::shared_ptr<ov::Model>& model, MoEStructureInfo& info) {
    LOG_DEBUG("Detecting router scores parameter from output path...");

    for (const auto& result_node : model->get_results()) {
        auto current = result_node->input_value(0).get_node_shared_ptr();

        // Skip Convert node if present
        if (auto convert_node = std::dynamic_pointer_cast<ov::op::v0::Convert>(current)) {
            LOG_DEBUG("  Skipping Convert node before Result");
            current = convert_node->input_value(0).get_node_shared_ptr();
        }

        // Skip ReduceSum node if present
        if (auto reduce_sum_node = std::dynamic_pointer_cast<ov::op::v1::ReduceSum>(current)) {
            LOG_DEBUG("  Skipping ReduceSum node before Result");
            current = reduce_sum_node->input_value(0).get_node_shared_ptr();
        }

        // Check for Multiply node
        auto multiply_node = std::dynamic_pointer_cast<ov::op::v1::Multiply>(current);
        if (!multiply_node)
            continue;

        LOG_DEBUG("  Found Multiply node before Result");
        info.router_scores_multiply_node = multiply_node;

        // Multiply has two inputs, one is expert output (from Reshape), other is router scores
        for (size_t i = 0; i < 2; ++i) {
            auto multiply_input = multiply_node->input_value(i).get_node_shared_ptr();

            // Skip the Reshape input (expert output)
            if (std::dynamic_pointer_cast<ov::op::v1::Reshape>(multiply_input)) {
                continue;
            }

            // Trace back to find Parameter (router scores)
            info.router_scores_idx = trace_to_parameter(multiply_input, model);

            if (info.router_scores_idx.has_value()) {
                LOG_DEBUG("  Found router scores parameter at index " << info.router_scores_idx.value());
                return true;
            }
        }
    }

    LOG_WARN("Could not find router scores Multiply node in output path");
    return false;
}

// Analyze MoE model structure - comprehensive single-pass analysis
std::optional<MoEStructureInfo> analyze_moe_structure(const std::shared_ptr<ov::Model>& model) {
    LOG_DEBUG("Analyzing MoE model structure...");
    LOG_BLOCK();

    MoEStructureInfo info;

    // Step 1: Find Tile operation and extract expert configuration
    if (!find_tile_and_extract_config(model, info)) {
        return std::nullopt;
    }

    // Step 2: Detect router_scores parameter from output path
    if (!find_router_scores_from_output(model, info)) {
        return std::nullopt;
    }

    // Step 3: Infer processing mode based on input_token_count
    // EXPERT_BATCH: token_count == 1 (single token, K experts in parallel)
    // EXPERT_ITERATIVE: token_count > 1 (multiple tokens, iterate through experts)
    if (info.input_token_count == 1) {
        info.processing_mode = moe::MoEProcessingMode::EXPERT_BATCH;
        LOG_DEBUG("Inferred EXPERT_BATCH mode (input_token_count=1)");
    } else if (info.input_token_count > 1) {
        info.processing_mode = moe::MoEProcessingMode::EXPERT_ITERATIVE;
        LOG_DEBUG("Inferred EXPERT_ITERATIVE mode (input_token_count=" << info.input_token_count << ")");
    }

    LOG_INFO("MoE structure analysis completed:");
    LOG_INFO("  - Num experts: " << info.num_experts);
    LOG_INFO("  - Expert hidden dim: " << info.expert_hidden_dim);
    LOG_INFO("  - Input token count: " << info.input_token_count);
    LOG_INFO("  - Mode: " << (info.is_expert_batch_mode() ? "EXPERT_BATCH" : "EXPERT_ITERATIVE"));
    if (info.expert_input_param_idx.has_value()) {
        LOG_INFO("  - Expert input param idx: " << info.expert_input_param_idx.value());
    }
    if (info.router_scores_idx.has_value()) {
        LOG_INFO("  - Router scores param idx: " << info.router_scores_idx.value());
    }

    return info;
}

// Determine transformation parameters based on structure analysis and router K
MoETransformConfig determine_transformation_params(const MoEStructureInfo& structure_info,
                                                   size_t k_from_router,
                                                   size_t iterative_chunk_size) {
    LOG_DEBUG("Determining transformation parameters...");
    LOG_BLOCK();

    MoETransformConfig config;

    if (structure_info.is_expert_batch_mode()) {
        // EXPERT_BATCH mode: transform to K active experts for parallel processing
        config.num_target_experts = k_from_router;
        config.chunk_size = 0;  // Not used in batch mode

        LOG_INFO("Transformation config for EXPERT_BATCH mode:");
        LOG_INFO("  - Mode: BATCH_K_EXPERTS");
        LOG_INFO("  - Target experts: " << config.num_target_experts);
    } else {
        // EXPERT_ITERATIVE mode: transform to 1 expert for iterative processing
        config.num_target_experts = 1;
        config.chunk_size = iterative_chunk_size;

        LOG_INFO("Transformation config for EXPERT_ITERATIVE mode:");
        LOG_INFO("  - Mode: SINGLE_EXPERT_ITERATIVE");
        LOG_INFO("  - Target experts: 1");
        LOG_INFO("  - Chunk size: " << config.chunk_size);
    }

    return config;
}

// Helper: Update Reshape node's constant input at a specific dimension
// MoE Reshape patterns: 3D [num_experts, token_num, hidden_dim] or 4D [num_experts, 1, token_num, hidden_dim]
// - dimension_index can be:
//   * 0: for num_experts (first dimension)
//   * -2: for token_num (second-to-last dimension, works for both 3D and 4D)
static bool update_reshape_constant_dimension(const std::shared_ptr<ov::op::v1::Reshape>& reshape_node,
                                              int dimension_index,
                                              size_t old_value,
                                              size_t new_value) {
    auto shape_input = reshape_node->input_value(1);
    auto shape_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(shape_input.get_node_shared_ptr());
    if (!shape_const) {
        return false;
    }

    auto shape_data = shape_const->cast_vector<int64_t>();

    // MoE reshapes are either 3D or 4D
    if (shape_data.size() < 3 || shape_data.size() > 4) {
        return false;
    }

    // Convert negative index to positive (e.g., -2 means second-to-last)
    size_t actual_index;
    if (dimension_index < 0) {
        actual_index = shape_data.size() + dimension_index;
    } else {
        actual_index = dimension_index;
    }

    // Check if the dimension value matches and update it
    if (actual_index < shape_data.size() && shape_data[actual_index] == static_cast<int64_t>(old_value)) {
        LOG_DEBUG("  Updating Reshape '" << reshape_node->get_friendly_name() << "' shape[" << actual_index << "] from "
                                         << old_value << " to " << new_value);

        shape_data[actual_index] = new_value;

        auto new_shape_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{shape_data.size()}, shape_data);
        reshape_node->input(1).replace_source_output(new_shape_const->output(0));

        return true;
    }

    return false;
}

// Helper: Check if parameter matches downstream pattern
// Pattern: Parameter -> [Convert] -> ReduceSum (Convert is optional)
static bool check_downstream_pattern(const std::shared_ptr<ov::op::v0::Parameter>& param) {
    // Parameter should have exactly one user
    auto param_users = param->output(0).get_target_inputs();
    if (param_users.size() != 1) {
        return false;
    }

    auto first_consumer = param_users.begin()->get_node()->shared_from_this();

    // Check if first consumer is ReduceSum (no Convert)
    if (auto reduce_sum = std::dynamic_pointer_cast<ov::op::v1::ReduceSum>(first_consumer)) {
        return true;
    }

    // Check if first consumer is Convert
    auto convert_node = std::dynamic_pointer_cast<ov::op::v0::Convert>(first_consumer);
    if (!convert_node) {
        return false;
    }

    // Convert should have exactly one user (ReduceSum node)
    auto convert_users = convert_node->output(0).get_target_inputs();
    if (convert_users.size() != 1) {
        return false;
    }

    auto reduce_sum =
        std::dynamic_pointer_cast<ov::op::v1::ReduceSum>(convert_users.begin()->get_node()->shared_from_this());
    return reduce_sum != nullptr;
}

std::optional<MoEDownstream> detect_and_transform_moe_downstream(const std::shared_ptr<ov::Model>& model,
                                                                 size_t active_experts_num) {
    LOG_DEBUG("Detecting MoE downstream pattern...");
    LOG_BLOCK();

    // Pattern to match: Parameter -> Convert -> ReduceSum
    // Looking for a parameter with shape [N, 1, H, W] where N is total_experts_num
    const auto& params = model->get_parameters();

    for (size_t param_idx = 0; param_idx < params.size(); ++param_idx) {
        const auto& param = params[param_idx];

        // Validate shape: must be [N, 1, H, W] where N > 1
        auto param_shape = param->get_partial_shape();
        if (!param_shape.rank().is_static() || param_shape.rank().get_length() != 4) {
            continue;
        }

        auto shape = param_shape.to_shape();
        if (shape[1] != 1 || shape[0] <= 1) {
            continue;
        }

        // Check pattern: Parameter -> Convert -> ReduceSum
        if (!check_downstream_pattern(param)) {
            continue;
        }

        LOG_DEBUG("  Found downstream pattern for Parameter: " << param->get_friendly_name());
        LOG_DEBUG("  Pattern matched: Parameter -> Convert -> ReduceSum");

        size_t total_experts_num = shape[0];

        // Validate configuration
        if (active_experts_num > total_experts_num) {
            LOG_WARN("Active experts (" << active_experts_num << ") > total experts (" << total_experts_num
                                        << "), skipping this parameter");
            continue;
        }

        // Clone and modify the model
        auto modified_model = model->clone();
        auto& cloned_params = modified_model->get_parameters();
        auto& target_param = cloned_params[param_idx];

        auto new_shape = shape;
        new_shape[0] = active_experts_num;

        LOG_DEBUG("  Modifying parameter shape from " << shape << " to " << new_shape);
        LOG_DEBUG("  Parameter name: " << target_param->get_friendly_name());

        target_param->set_partial_shape(ov::PartialShape(new_shape));
        target_param->validate_and_infer_types();
        modified_model->validate_nodes_and_infer_types();

        LOG_INFO("Successfully detected and transformed MoE downstream pattern");
        LOG_INFO("  Expert output parameter: " << param->get_friendly_name());
        LOG_INFO("  Expert output parameter index: " << param_idx);

        // Build and return result
        MoEDownstream result;
        result.total_experts_num = total_experts_num;
        result.active_experts_num = active_experts_num;
        result.expert_output_param_idx = param_idx;
        result.modified_model = modified_model;

        return result;
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

// ============================================================================
// MoEModelTransformer Implementation
// ============================================================================

std::optional<MoEModelTransformer::KeyNodes> MoEModelTransformer::find_key_nodes_in_cloned_model(
    const std::shared_ptr<ov::Model>& model) const {
    KeyNodes nodes;
    const std::string tile_name = m_structure_info.expert_input_tile_node->get_friendly_name();
    const std::string multiply_name = m_structure_info.router_scores_multiply_node->get_friendly_name();

    for (const auto& node : model->get_ordered_ops()) {
        if (auto tile = std::dynamic_pointer_cast<ov::op::v0::Tile>(node)) {
            if (tile->get_friendly_name() == tile_name) {
                nodes.tile_node = tile;
                LOG_DEBUG("Found Tile node in cloned model: " << tile_name);
            }
        }

        if (auto multiply = std::dynamic_pointer_cast<ov::op::v1::Multiply>(node)) {
            if (multiply->get_friendly_name() == multiply_name) {
                nodes.multiply_node = multiply;
                LOG_DEBUG("Found Multiply node in cloned model: " << multiply_name);
            }
        }

        if (nodes.tile_node && nodes.multiply_node) {
            break;  // Found all required nodes
        }
    }

    if (!nodes.tile_node || !nodes.multiply_node) {
        LOG_ERROR("Could not find required nodes in cloned model - Tile: " << tile_name
                                                                           << ", Multiply: " << multiply_name);
        return std::nullopt;
    }

    return nodes;
}

std::pair<std::shared_ptr<ov::Node>, ov::Output<ov::Node>> MoEModelTransformer::find_matmul_and_predecessor(
    ov::Output<ov::Node>& tile_output) const {
    for (const auto& target_input : tile_output.get_target_inputs()) {
        auto consumer = target_input.get_node()->shared_from_this();

        if (auto reshape_op = std::dynamic_pointer_cast<ov::op::v1::Reshape>(consumer)) {
            LOG_DEBUG("Found Reshape after Tile: " << reshape_op->get_friendly_name());
            for (const auto& reshape_target : reshape_op->output(0).get_target_inputs()) {
                if (auto mm =
                        std::dynamic_pointer_cast<ov::op::v0::MatMul>(reshape_target.get_node()->shared_from_this())) {
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
}

void MoEModelTransformer::replace_tile_with_new_tile(const ov::Output<ov::Node>& tile_input,
                                                     ov::Output<ov::Node>& node_before_matmul,
                                                     const std::string& friendly_name,
                                                     size_t num_target_experts) const {
    auto matmul_input_shape = node_before_matmul.get_shape();
    ov::Shape target_expert_shape = matmul_input_shape;
    target_expert_shape[0] = num_target_experts;

    LOG_DEBUG("Original MatMul input shape: " << matmul_input_shape);
    LOG_DEBUG("Target expert shape (" << num_target_experts << " experts): " << target_expert_shape);

    std::shared_ptr<ov::Node> new_node;

    if (num_target_experts == 1) {
        // For single expert (iterative mode), use Reshape since no repetition needed
        auto target_shape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                         ov::Shape{target_expert_shape.size()},
                                                                         target_expert_shape);
        new_node = std::make_shared<ov::op::v1::Reshape>(tile_input, target_shape_const, false);
        new_node->set_friendly_name(friendly_name + "_transformed_single_expert");
        LOG_DEBUG("Using Reshape for single expert");
    } else {
        // For multiple active experts (batch mode), use Tile to replicate data, then Reshape to expand dims
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
}

void MoEModelTransformer::ensure_tensor_names(const std::shared_ptr<ov::Model>& model) const {
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
}

void MoEModelTransformer::fix_parameters_with_num_experts(const std::shared_ptr<ov::Model>& model,
                                                          size_t num_experts,
                                                          size_t num_target_experts) const {
    LOG_DEBUG("Fixing Parameter nodes with num_experts in shape to " << num_target_experts << " experts...");

    const auto& all_params = model->get_parameters();
    for (size_t param_idx = 0; param_idx < all_params.size(); ++param_idx) {
        const auto& param = all_params[param_idx];

        // Skip the tile input parameter - it does not contain expert dimension
        if (m_structure_info.expert_input_param_idx.has_value() &&
            param_idx == m_structure_info.expert_input_param_idx.value()) {
            LOG_DEBUG("  Skipping tile input parameter at index " << param_idx << ": " << param->get_friendly_name());
            continue;
        }

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

                shape[fix_dim] = num_target_experts;

                // Directly set the new shape on the parameter
                param->set_partial_shape(ov::PartialShape(shape));
                param->validate_and_infer_types();
                LOG_DEBUG("    Successfully updated parameter shape to " << num_target_experts);
            }
        }
    }

    // Also fix Reshape nodes that contain num_experts in their shape constants
    LOG_DEBUG("Fixing Reshape nodes with num_experts in shape...");
    for (const auto& node : model->get_ordered_ops()) {
        auto reshape_node = std::dynamic_pointer_cast<ov::op::v1::Reshape>(node);
        if (!reshape_node) {
            continue;
        }

        // For MoE Reshape patterns (3D or 4D), num_experts is always at dimension 0
        // 3D: [num_experts, token_num, hidden_dim]
        // 4D: [num_experts, 1, token_num, hidden_dim]
        update_reshape_constant_dimension(reshape_node, 0, num_experts, num_target_experts);
    }
}

void MoEModelTransformer::fix_token_count_for_expert_iterative(
    const std::shared_ptr<ov::Model>& model,
    size_t num_target_experts,
    size_t chunk_size,
    const std::shared_ptr<ov::op::v0::Tile>& expert_input_tile_op,
    const std::shared_ptr<ov::op::v1::Multiply>& router_scores_multiply_op) const {
    if (num_target_experts != 1) {
        return;  // Only apply to EXPERT_ITERATIVE mode (single expert)
    }

    LOG_DEBUG("Fixing token count from " << m_structure_info.input_token_count << " to " << chunk_size
                                         << " for EXPERT_ITERATIVE mode...");

    if (!expert_input_tile_op) {
        LOG_WARN("Tile node not available for token count fixing");
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

    trace_to_params(expert_input_tile_op->input_value(0));

    // Also find router parameter from the Multiply node
    LOG_DEBUG("Tracing router parameter from Multiply node");
    for (size_t i = 0; i < 2; ++i) {
        auto multiply_input = router_scores_multiply_op->input_value(i).get_node_shared_ptr();

        // Skip Reshape input (that's the expert output)
        if (std::dynamic_pointer_cast<ov::op::v1::Reshape>(multiply_input)) {
            continue;
        }

        trace_to_params(router_scores_multiply_op->input_value(i));
    }

    // Fix Parameter shapes
    size_t original_token_count = m_structure_info.input_token_count;
    for (const auto& param : params_to_fix) {
        auto param_shape = param->get_partial_shape();
        if (param_shape.rank().is_static() && param_shape.rank().get_length() >= 2) {
            auto shape = param_shape.to_shape();

            // Find dimension with original token count
            for (size_t i = 0; i < shape.size(); ++i) {
                if (shape[i] == original_token_count) {
                    LOG_DEBUG("  Updating Parameter '" << param->get_friendly_name() << "' shape[" << i << "] from "
                                                       << original_token_count << " to " << chunk_size);
                    shape[i] = chunk_size;
                }
            }

            param->set_partial_shape(ov::PartialShape(shape));
            param->validate_and_infer_types();
        }
    }

    // Also fix Reshape nodes that contain original_token_count in their shape constants
    LOG_DEBUG("Fixing Reshape nodes with token_count in shape...");
    for (const auto& node : model->get_ordered_ops()) {
        auto reshape_node = std::dynamic_pointer_cast<ov::op::v1::Reshape>(node);
        if (!reshape_node) {
            continue;
        }

        // For MoE Reshape patterns (3D or 4D), token_num is always at second-to-last dimension
        // 3D: [num_experts, token_num, hidden_dim] - token_num at index 1
        // 4D: [num_experts, 1, token_num, hidden_dim] - token_num at index 2
        // Use -2 to refer to second-to-last dimension in both cases
        update_reshape_constant_dimension(reshape_node, -2, original_token_count, chunk_size);
    }

    // Trigger shape inference to propagate changes through the model
    LOG_DEBUG("Triggering shape inference after token count changes...");
    model->validate_nodes_and_infer_types();
}

// Unroll MoE expert model on expert dimension using GraphRewrite patterns
std::shared_ptr<ov::Model> MoEModelTransformer::unroll_expert_dimension(const std::shared_ptr<ov::Model>& model,
                                                                        size_t num_experts,
                                                                        bool full_optimization) const {
    if (num_experts <= 1) {
        LOG_DEBUG("No unrolling needed for single expert");
        return model;
    }

    LOG_INFO("Unrolling expert dimension for " << num_experts << " experts using GraphRewrite");
    LOG_INFO("Optimization mode: " << (full_optimization ? "Full (weights + activations)" : "WeightsOnly"));
    LOG_BLOCK();

    try {
        auto unrolled_model = model->clone();

        ov::pass::Manager manager;
        if (full_optimization) {
            manager.register_pass<ov::npuw::pass::MoEExpertUnrolling>(unrolled_model);
        } else {
            // Weight-only optimizations
            manager.register_pass<ov::npuw::pass::MoEExpertUnrollingWeightsOnly>(unrolled_model);
            // AWQ parameter multiply unrolling with cleanup
            manager.register_pass<ov::npuw::pass::UnrollAWQParameterMultiply>(unrolled_model);
        }

        manager.run_passes(unrolled_model);

        unrolled_model->validate_nodes_and_infer_types();
        LOG_INFO("Successfully unrolled and validated model for " << num_experts << " experts");

        return unrolled_model;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to unroll expert dimension: " << e.what());
        return nullptr;
    }
}

// Transform MoE model based on configuration
std::shared_ptr<ov::Model> MoEModelTransformer::apply_expert_transformation(
    const std::shared_ptr<ov::Model>& original_model,
    const MoETransformConfig& config) const {
    LOG_DEBUG("Transforming MoE model to " << config.num_target_experts << " expert(s), mode: "
                                           << (config.is_expert_iterative() ? "EXPERT_ITERATIVE" : "EXPERT_BATCH")
                                           << ", chunk_size: " << config.chunk_size);
    LOG_BLOCK();

    auto model = original_model->clone();
    const auto num_experts = m_structure_info.num_experts;
    const auto num_target_experts = config.num_target_experts;
    const auto chunk_size = config.chunk_size;

    // Find key nodes in the cloned model
    auto key_nodes = find_key_nodes_in_cloned_model(model);
    if (!key_nodes) {
        return nullptr;
    }
    auto expert_input_tile_op = key_nodes->tile_node;
    auto router_scores_multiply_op = key_nodes->multiply_node;

    // Main transformation flow
    auto tile_input = expert_input_tile_op->input_value(0);
    auto tile_output = expert_input_tile_op->output(0);

    auto [matmul_node, node_before_matmul] = find_matmul_and_predecessor(tile_output);
    if (!matmul_node) {
        LOG_ERROR("Could not find MatMul operation after Tile");
        return nullptr;
    }

    // Apply transformations in sequence
    fix_token_count_for_expert_iterative(model,
                                         num_target_experts,
                                         chunk_size,
                                         expert_input_tile_op,
                                         router_scores_multiply_op);
    replace_tile_with_new_tile(tile_input,
                               node_before_matmul,
                               expert_input_tile_op->get_friendly_name(),
                               num_target_experts);
    fix_parameters_with_num_experts(model, num_experts, num_target_experts);
    ensure_tensor_names(model);

    model->validate_nodes_and_infer_types();
    LOG_DEBUG("Successfully transformed to " << num_target_experts << " expert(s) model");

    if (num_target_experts > 1) {
        model = unroll_expert_dimension(model, num_target_experts);
    }

    return model;
}

std::optional<MoEExperts> MoEExperts::from(const std::shared_ptr<ov::Model>& model,
                                           const std::shared_ptr<ov::Model>& router_model,
                                           size_t iterative_chunk_size) {
    LOG_DEBUG("Creating MoEExperts from model: " << model->get_friendly_name());
    LOG_BLOCK();

    // Step 1: Extract K from router model
    if (!router_model) {
        LOG_ERROR("Router model is required to extract K from TopK node");
        return std::nullopt;
    }

    auto k_from_router = extract_k_from_router(router_model);
    if (!k_from_router.has_value()) {
        LOG_ERROR("Failed to extract K from router model");
        return std::nullopt;
    }

    size_t k_value = k_from_router.value();
    LOG_INFO("Extracted K=" << k_value << " from router model");

    // Step 2: Analyze model structure
    auto structure_info = analyze_moe_structure(model);
    if (!structure_info || !structure_info->is_valid()) {
        LOG_WARN("Model structure analysis failed for MoE expert pattern");
        return std::nullopt;
    }

    // Step 3: Determine chunk sizes based on processing mode and configuration
    std::vector<size_t> chunk_sizes;
    if (structure_info->is_expert_batch_mode()) {
        // EXPERT_BATCH mode: single model with K active experts, no chunking
        chunk_sizes = {0};
        LOG_INFO("EXPERT_BATCH mode: Creating single model for K=" << k_value << " active experts");
    } else {
        if (iterative_chunk_size == 0) {
            chunk_sizes.assign(DEFAULT_ITERATIVE_CHUNKING_VARIANTS.begin(), DEFAULT_ITERATIVE_CHUNKING_VARIANTS.end());
            LOG_INFO("EXPERT_ITERATIVE mode: Creating multiple models for dynamic chunking");
        } else {
            chunk_sizes = {iterative_chunk_size};
            LOG_INFO("EXPERT_ITERATIVE mode: Creating single model with chunk_size=" << iterative_chunk_size);
        }
    }

    // Step 4: Transform models for each chunk size
    std::map<size_t, std::shared_ptr<ov::Model>> transformed_models;
    for (auto chunk_size : chunk_sizes) {
        auto config = determine_transformation_params(*structure_info, k_value, chunk_size);

        MoEModelTransformer transformer(*structure_info);
        auto transformed_model = transformer.apply_expert_transformation(model, config);
        if (!transformed_model) {
            LOG_WARN("Failed to transform model with chunk_size=" << chunk_size);
            continue;
        }

        // Store the transformed model
        transformed_models[chunk_size] = transformed_model;

        LOG_INFO("Successfully transformed model for chunk_size=" << chunk_size);
    }

    if (transformed_models.empty()) {
        LOG_ERROR("Failed to transform any models");
        return std::nullopt;
    }

    // Step 5: Build parameter mapping from any transformed model (all have same mapping)
    // Note: For EXPERT_ITERATIVE mode (single expert), no unrolling happens, so mapping will be empty
    //       For EXPERT_BATCH mode (K experts), unrolling creates the same mapping structure
    auto param_mapping = build_parameter_mapping_from_rtinfo(model, transformed_models.begin()->second);

    // Step 6: Populate and validate MoEExperts structure
    MoEExperts moe_experts;
    moe_experts._num_experts = structure_info->num_experts;
    moe_experts._expert_hidden_dim = structure_info->expert_hidden_dim;
    moe_experts._input_token_count = structure_info->input_token_count;
    moe_experts._chunk_token_count = structure_info->is_expert_batch_mode() ? 0 : iterative_chunk_size;
    moe_experts._num_active_experts = k_value;  // Store actual K from router
    moe_experts._transformed_models = std::move(transformed_models);
    moe_experts._router_scores_idx = structure_info->router_scores_idx;
    moe_experts._expert_input_param_idx = structure_info->expert_input_param_idx;
    moe_experts._param_mapping = std::move(param_mapping);

    if (!moe_experts.is_valid()) {
        LOG_ERROR("Created MoEExperts structure is invalid");
        return std::nullopt;
    }

    LOG_INFO("Successfully created MoEExperts:");
    LOG_INFO("  - Total number of experts: " << moe_experts._num_experts);
    LOG_INFO("  - Num active experts: " << moe_experts._num_active_experts);
    LOG_INFO("  - Expert hidden dim: " << moe_experts._expert_hidden_dim);
    LOG_INFO("  - Number of transformed models: " << moe_experts._transformed_models.size());
    if (structure_info->is_expert_batch_mode()) {
        LOG_INFO("    - EXPERT_BATCH model (no chunking)");
    } else {
        LOG_INFO("    - EXPERT_ITERATIVE models with chunk sizes:");
        for (const auto& entry : moe_experts._transformed_models) {
            LOG_INFO("      * chunk_size=" << entry.first);
        }
    }

    return moe_experts;
}

}  // namespace function

namespace compiled {

MoEExperts::MoEExperts(const function::MoEExperts& func_moe) {
    num_experts = func_moe._num_experts;
    expert_hidden_dim = func_moe._expert_hidden_dim;
    num_active_experts = func_moe._num_active_experts;
    input_token_count = func_moe._input_token_count;
    _models_to_compile = func_moe._transformed_models;
    _router_scores_idx = func_moe._router_scores_idx;
    _expert_input_param_idx = func_moe._expert_input_param_idx;
    _param_mapping = func_moe._param_mapping;

    LOG_DEBUG("Created compiled::MoEExperts:");
    LOG_DEBUG("  Mode: " << (func_moe.is_expert_iterative() ? "EXPERT_ITERATIVE" : "EXPERT_BATCH"));
    LOG_DEBUG("  Total experts: " << num_experts);
    LOG_DEBUG("  Active experts: " << num_active_experts);
    LOG_DEBUG("  Input token count: " << input_token_count);
    LOG_DEBUG("  Number of models to compile: " << _models_to_compile.size());
    for (const auto& entry : _models_to_compile) {
        LOG_DEBUG("    Chunk size: " << entry.first);
    }
    if (_router_scores_idx.has_value()) {
        LOG_DEBUG("  Router scores parameter index: " << _router_scores_idx.value());
    }
    if (_expert_input_param_idx.has_value()) {
        LOG_DEBUG("  Expert input parameter index: " << _expert_input_param_idx.value());
    }
}

void MoEExperts::set_compiled_model(size_t chunk_size, ov::SoPtr<ov::ICompiledModel>&& compiled_model) {
    _compiled_models[chunk_size] = std::move(compiled_model);
    _models_to_compile.erase(chunk_size);  // Free memory after compilation

    LOG_DEBUG("Set compiled model for MoE experts, chunk_size=" << chunk_size);
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
}  // namespace npuw
}  // namespace ov
