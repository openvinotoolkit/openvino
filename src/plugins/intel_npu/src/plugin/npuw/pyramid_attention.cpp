// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pyramid_attention.hpp"

#include <algorithm>
#include <iostream>

#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/validate.hpp"
#include "util.hpp"

namespace opp = ov::pass::pattern;

namespace ov {
namespace npuw {
namespace function {

// Helper function to find mask parameter by traversing from Add node
std::shared_ptr<ov::op::v0::Parameter> find_mask_parameter(const std::shared_ptr<ov::Node>& add_node) {
    if (!add_node || add_node->get_input_size() < 2) {
        return nullptr;
    }

    // Traverse the Add node's mask input (input 1) upwards to find the proper Parameter
    // Only unary ops are allowed along the way
    auto mask_in_node = add_node->input(1).get_source_output().get_node_shared_ptr();
    while (mask_in_node && !ov::op::util::is_parameter(mask_in_node)) {
        if (mask_in_node->inputs().size() != 1) {
            LOG_WARN("Non-unary or disconnected op on the way from Add to input mask");
            return nullptr;
        }
        mask_in_node = mask_in_node->inputs()[0].get_source_output().get_node_shared_ptr();
    }

    if (mask_in_node && ov::op::util::is_parameter(mask_in_node)) {
        return std::static_pointer_cast<ov::op::v0::Parameter>(mask_in_node);
    }

    return nullptr;
}

// Helper function to patch broadcast constants (set to 1 for dynamic handling)
void patch_broadcast_constants(const std::shared_ptr<ov::Model>& model, size_t target_length) {
    for (auto&& op : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v3::Broadcast>(op)) {
            continue;
        }
        // Inspect the constant
        auto shape_source = op->input(1).get_source_output().get_node_shared_ptr();
        if (!ov::is_type<ov::op::v0::Constant>(shape_source)) {
            LOG_WARN("SDPA Broadcast's 2nd input is not Const: " << shape_source << ", skipping");
            continue;
        }

        auto shape_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(shape_source);
        auto shape_values = shape_const->cast_vector<int32_t>();
        for (auto&& d : shape_values) {
            //  Assume the context length is the mask's innermost dimension
            if (static_cast<std::size_t>(d) == target_length) {
                d = 1;
            }
        }
        auto new_const = std::make_shared<ov::op::v0::Constant>(shape_const->get_element_type(),
                                                                shape_const->get_shape(),
                                                                shape_values);
        op->input(1).replace_source_output(new_const);
    }
}

// Helper function to patch reshape constants for pre-reshape (-1 substitution)
void patch_reshape_constants_pre_reshape(const std::shared_ptr<ov::Model>& model,
                                         const std::map<std::string, size_t>& past_value_sequence_dims) {
    for (auto&& op : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v1::Reshape>(op)) {
            continue;
        }

        // Check if Reshape's single consumer is MatMul
        auto target_inputs = op->output(0).get_target_inputs();
        if (target_inputs.size() != 1) {
            continue;  // Reshape should have exactly one consumer
        }

        auto matmul_node = target_inputs.begin()->get_node()->shared_from_this();
        if (!ov::is_type<ov::op::v0::MatMul>(matmul_node)) {
            continue;
        }

        // Check if MatMul's input 0 is from Softmax
        auto matmul_input0 = matmul_node->input(0).get_source_output().get_node_shared_ptr();
        if (!ov::is_type<ov::op::v8::Softmax>(matmul_input0)) {
            continue;
        }

        LOG_INFO("Found Reshape -> MatMul pattern where MatMul input 0 is from Softmax, patching Reshape constant");

        // Inspect the reshape constant (shape input)
        auto shape_source = op->input(1).get_source_output().get_node_shared_ptr();
        if (!ov::is_type<ov::op::v0::Constant>(shape_source)) {
            LOG_WARN("Reshape's shape input is not Const: " << shape_source << ", skipping");
            continue;
        }

        auto shape_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(shape_source);
        auto shape_values = shape_const->cast_vector<int32_t>();

        // Find the first past value sequence dimension from the map
        // All past value parameters should have the same sequence dimension
        if (past_value_sequence_dims.empty()) {
            LOG_WARN("No past value sequence dimensions provided for reshape patching");
            continue;
        }

        size_t value_seq_dim = past_value_sequence_dims.begin()->second;
        NPUW_ASSERT(value_seq_dim < shape_values.size());
        shape_values[value_seq_dim] = -1;

        auto new_const = std::make_shared<ov::op::v0::Constant>(shape_const->get_element_type(),
                                                                shape_const->get_shape(),
                                                                shape_values);
        op->input(1).replace_source_output(new_const);
    }
}

// Helper function to create Attention instance from a model
std::optional<ov::npuw::function::Attention> create_attention_from_model(
    const std::shared_ptr<ov::Model>& model,
    const std::map<std::string, size_t>& past_key_sequence_dims,
    const std::map<std::string, size_t>& past_value_sequence_dims) {
    // Find SDPA pattern nodes in the model
    auto pattern_nodes = find_sdpa_pattern_nodes(model);
    if (!pattern_nodes.is_valid()) {
        LOG_WARN("Could not find SDPA pattern in model");
        return std::nullopt;
    }

    // Find mask parameter in the model
    auto mask_param = find_mask_parameter(pattern_nodes.add_node);
    if (!mask_param) {
        LOG_WARN("Could not find mask parameter in model");
        return std::nullopt;
    }

    // Create Attention instance
    ov::npuw::function::Attention attention;
    attention._mask = mask_param;
    attention._mask_shape = mask_param->get_shape();

    // Add past key/value inputs to attention
    const auto& params = model->get_parameters();
    for (const auto& param : params) {
        const std::string param_name = param->get_friendly_name();
        if (ov::npuw::util::isPastKeyValuesKey(param_name)) {
            auto dim_iter = past_key_sequence_dims.find(param_name);
            if (dim_iter != past_key_sequence_dims.end()) {
                attention._inputs.push_back(ov::npuw::function::Attention::Param{param, dim_iter->second});
            }
        } else if (ov::npuw::util::isPastKeyValuesValue(param_name)) {
            auto dim_iter = past_value_sequence_dims.find(param_name);
            if (dim_iter != past_value_sequence_dims.end()) {
                attention._inputs.push_back(ov::npuw::function::Attention::Param{param, dim_iter->second});
            }
        }
    }

    return attention;
}

// Helper function to process a single pyramid model (clone, reshape, patch, optimize)
std::optional<PyramidModelResult> process_pyramid_model(const std::shared_ptr<ov::Model>& original_model,
                                                        size_t model_idx,
                                                        size_t query_length,
                                                        size_t full_context_length,
                                                        const std::map<std::string, size_t>& past_key_sequence_dims,
                                                        const std::map<std::string, size_t>& past_value_sequence_dims) {
    // Clone the original model for modification
    auto cloned_model = original_model->clone();

    // Calculate dimensions for this model
    size_t current_context_length = (model_idx + 1) * query_length;
    size_t current_past_length = model_idx * query_length;

    LOG_DEBUG("Model " << model_idx << ":");
    LOG_DEBUG("  Context length: " << current_context_length);
    LOG_DEBUG("  Past length: " << current_past_length);

    // Create initial Attention instance to get mask parameter for reshaping
    auto initial_attention =
        create_attention_from_model(cloned_model, past_key_sequence_dims, past_value_sequence_dims);
    if (!initial_attention) {
        LOG_WARN("Could not create attention from cloned model " << model_idx);
        return std::nullopt;
    }

    auto cloned_mask_param = initial_attention->_mask;

    // Create reshape map for this model
    std::map<ov::Output<ov::Node>, ov::PartialShape> new_shapes;

    // Update parameters shapes
    const auto& params = cloned_model->get_parameters();
    for (auto&& param : params) {
        const std::string param_name = param->get_friendly_name();
        auto original_shape = param->get_shape();
        ov::PartialShape new_shape = original_shape;

        // Handle attention mask parameter - use the mask parameter found in cloned model
        if (param == cloned_mask_param) {
            // Update the last dimension to current context length
            if (new_shape.size() >= 1) {
                new_shape[new_shape.size() - 1] = current_context_length;
                new_shapes[param->output(0)] = new_shape;
                LOG_DEBUG("  Mask param '" << param_name << "' shape: " << original_shape << " -> " << new_shape);
            }
        }
        // Handle past key parameters
        else if (ov::npuw::util::isPastKeyValuesKey(param_name)) {
            // Use pre-analyzed sequence dimension information
            auto dim_iter = past_key_sequence_dims.find(param_name);
            if (dim_iter != past_key_sequence_dims.end()) {
                size_t sequence_dim_idx = dim_iter->second;
                new_shape[sequence_dim_idx] = current_past_length;
                new_shapes[param->output(0)] = new_shape;
                LOG_DEBUG("  Past key param '" << param_name << "' shape: " << original_shape << " -> " << new_shape);

                // Record past key input (will be handled later)
            } else {
                LOG_WARN("No pre-analyzed sequence dimension for past key param: " << param_name);
                return std::nullopt;
            }
        }
        // Handle past value parameters
        else if (ov::npuw::util::isPastKeyValuesValue(param_name)) {
            // Use pre-analyzed sequence dimension information
            auto dim_iter = past_value_sequence_dims.find(param_name);
            if (dim_iter != past_value_sequence_dims.end()) {
                size_t sequence_dim_idx = dim_iter->second;
                new_shape[sequence_dim_idx] = current_past_length;
                new_shapes[param->output(0)] = new_shape;
                LOG_DEBUG("  Past value param '" << param_name << "' shape: " << original_shape << " -> " << new_shape);

                // Record past value input (will be handled later)
            } else {
                LOG_WARN("No pre-analyzed sequence dimension for past value param: " << param_name);
                return std::nullopt;
            }
        }
    }

    // Apply the reshaping to the cloned model
    if (new_shapes.empty()) {
        LOG_WARN("No parameters found for reshaping in model " << model_idx << ", skipping this model");
        return std::nullopt;
    }

    // Apply pre-reshape patching using helper functions
    patch_broadcast_constants(cloned_model, full_context_length);
    patch_reshape_constants_pre_reshape(cloned_model, past_value_sequence_dims);

    cloned_model->reshape(new_shapes);
    cloned_model->validate_nodes_and_infer_types();

    LOG_DEBUG("Model " << model_idx << " reshaped successfully");

    // Create final Attention instance after reshape
    auto final_attention = create_attention_from_model(cloned_model, past_key_sequence_dims, past_value_sequence_dims);
    if (!final_attention) {
        LOG_WARN("Could not create final attention after reshape for model " << model_idx);
        return std::nullopt;
    }

    LOG_DEBUG("  Updated attention after reshape");

    // Log attention information for this model
    LOG_DEBUG("  Attention info - mask: " << (final_attention->_mask ? "present" : "absent"));
    LOG_DEBUG("  Attention info - inputs count: " << final_attention->_inputs.size());
    if (final_attention->_mask) {
        LOG_DEBUG("  Mask shape: " << final_attention->_mask_shape);
    }

    return PyramidModelResult{cloned_model, std::move(*final_attention)};
}

// Helper function to validate model and extract necessary information for pyramid attention
std::optional<PyramidValidationResult> validate_and_setup_pyramid_attention(const std::shared_ptr<ov::Model>& model) {
    // Find SDPA pattern nodes using the extracted function
    auto pattern_nodes = find_sdpa_pattern_nodes(model);
    if (!pattern_nodes.is_valid()) {
        LOG_WARN("Could not find valid SDPA pattern in model");
        return std::nullopt;
    }

    LOG_INFO("Found SDPA pattern: MatMul -> Add -> Softmax -> MatMul");

    // Extract query_length and full_context_length from Softmax output shape
    auto softmax_output_shape = pattern_nodes.softmax_node->get_output_shape(0);
    size_t query_length = 0;
    size_t full_context_length = 0;

    if (softmax_output_shape.size() >= 2) {
        full_context_length = softmax_output_shape.back();                     // Last dimension
        query_length = softmax_output_shape[softmax_output_shape.size() - 2];  // Second-to-last dimension

        LOG_DEBUG("Extracted from Softmax output shape:");
        LOG_DEBUG("  Query length: " << query_length);
        LOG_DEBUG("  Full context length: " << full_context_length);
    } else {
        LOG_WARN("Softmax output shape has insufficient dimensions: " << softmax_output_shape.size());
        return std::nullopt;
    }

    // Early return for invalid parameters
    if (query_length == 0 || full_context_length == 0 || full_context_length < query_length) {
        LOG_WARN("Invalid query_length (" << query_length << ") or full_context_length (" << full_context_length
                                          << ") for pyramid attention");
        return std::nullopt;
    }

    // Pre-analyze original model to find sequence dimensions for past key/value parameters
    // This avoids repeated analysis in each cloned model
    std::map<std::string, size_t> past_key_sequence_dims;
    std::map<std::string, size_t> past_value_sequence_dims;

    // Helper function to find sequence dimension in parameter shape
    auto find_context_dim = [](const std::shared_ptr<ov::op::v0::Parameter>& param,
                               size_t target_length) -> std::optional<size_t> {
        const auto& param_shape = param->get_shape();
        auto dim_iter = std::find(param_shape.begin(), param_shape.end(), target_length);
        if (dim_iter == param_shape.end()) {
            return std::nullopt;  // No such dim found
        }
        if (std::find(dim_iter + 1, param_shape.end(), target_length) != param_shape.end()) {
            return std::nullopt;  // There must be no other such dim
        }
        return std::distance(param_shape.begin(), dim_iter);
    };

    // Analyze original model parameters to find sequence dimensions
    const auto& original_params = model->get_parameters();
    for (const auto& param : original_params) {
        const std::string param_name = param->get_friendly_name();

        if (ov::npuw::util::isPastKeyValuesKey(param_name)) {
            auto sequence_dim_opt = find_context_dim(param, full_context_length - query_length);
            if (sequence_dim_opt) {
                past_key_sequence_dims[param_name] = *sequence_dim_opt;
                LOG_DEBUG("Found past key sequence dimension for '" << param_name << "': " << *sequence_dim_opt);
            } else {
                LOG_WARN("Could not find sequence dimension for past key param: " << param_name);
                return std::nullopt;
            }
        } else if (ov::npuw::util::isPastKeyValuesValue(param_name)) {
            auto sequence_dim_opt = find_context_dim(param, full_context_length - query_length);
            if (sequence_dim_opt) {
                past_value_sequence_dims[param_name] = *sequence_dim_opt;
                LOG_DEBUG("Found past value sequence dimension for '" << param_name << "': " << *sequence_dim_opt);
            } else {
                LOG_WARN("Could not find sequence dimension for past value param: " << param_name);
                return std::nullopt;
            }
        }
    }

    return PyramidValidationResult{query_length, full_context_length, past_key_sequence_dims, past_value_sequence_dims};
}

SDPAPatternNodes find_sdpa_pattern_nodes(const std::shared_ptr<ov::Model>& model) {
    // Find decomposed SDPA pattern components
    SDPAPatternNodes pattern_nodes;

    // Search for the pattern: MatMul -> Add -> Softmax -> MatMul
    auto ops = model->get_ordered_ops();
    for (auto&& node : ops) {
        if (ov::is_type<ov::op::v8::Softmax>(node)) {
            pattern_nodes.softmax_node = node;

            // Check if softmax is fed by Add
            auto softmax_input = node->input(0).get_source_output().get_node_shared_ptr();
            if (ov::is_type<ov::op::v1::Add>(softmax_input)) {
                pattern_nodes.add_node = softmax_input;

                // Check if add is fed by MatMul (first MatMul)
                auto add_input0 = pattern_nodes.add_node->input(0).get_source_output().get_node_shared_ptr();
                if (ov::is_type<ov::op::v0::MatMul>(add_input0)) {
                    pattern_nodes.matmul1_node = add_input0;
                }
            }

            // Check if softmax feeds into MatMul (second MatMul)
            for (auto&& output : node->outputs()) {
                for (auto&& target_input : output.get_target_inputs()) {
                    auto target_node = target_input.get_node()->shared_from_this();
                    if (ov::is_type<ov::op::v0::MatMul>(target_node)) {
                        pattern_nodes.matmul2_node = target_node;
                        break;
                    }
                }
                if (pattern_nodes.matmul2_node)
                    break;
            }

            if (pattern_nodes.is_valid()) {
                pattern_nodes.log_pattern();
                break;  // Found complete pattern
            }
        }
    }

    return pattern_nodes;
}

std::optional<PyramidAttention> PyramidAttention::from(const std::shared_ptr<ov::Model>& model) {
    // Validate and setup pyramid attention
    auto validation_result = validate_and_setup_pyramid_attention(model);
    if (!validation_result) {
        return std::nullopt;
    }

    size_t query_length = validation_result->query_length;
    size_t full_context_length = validation_result->full_context_length;
    const auto& past_key_sequence_dims = validation_result->past_key_sequence_dims;
    const auto& past_value_sequence_dims = validation_result->past_value_sequence_dims;

    std::vector<std::shared_ptr<ov::Model>> pyramid_models;
    size_t num_models = full_context_length / query_length;
    LOG_INFO("Creating " << num_models << " pyramid attention models");

    // Store Attention instances for each model
    std::vector<Attention> pyramid_attentions;

    for (size_t model_idx = 0; model_idx < num_models; ++model_idx) {
        // Optimization: The last model (num_models - 1) is the same as the original model
        // Skip reshape and directly use the original model pointer
        if (model_idx == num_models - 1) {
            LOG_INFO("Using original model for pyramid model[" << model_idx << "] (optimization)");
            pyramid_models.push_back(model);  // Direct use of original model pointer

            // Create Attention instance using the helper function
            auto last_attention = create_attention_from_model(model, past_key_sequence_dims, past_value_sequence_dims);
            if (!last_attention) {
                LOG_WARN("Could not create attention for original model");
                return std::nullopt;
            }

            pyramid_attentions.push_back(std::move(*last_attention));
            LOG_INFO("Successfully setup attention for original model[" << model_idx << "]");
        } else {
            // Process pyramid models 0 to num_models-2 using the helper function
            auto result = process_pyramid_model(model,
                                                model_idx,
                                                query_length,
                                                full_context_length,
                                                past_key_sequence_dims,
                                                past_value_sequence_dims);
            if (!result) {
                return std::nullopt;
            }

            pyramid_models.push_back(result->model);
            pyramid_attentions.push_back(std::move(result->attention));
        }
    }

    LOG_INFO("Successfully created " << pyramid_models.size() << " pyramid attention models");

    // Create PyramidAttention instance and set the extracted values
    PyramidAttention pyramid_attention;
    pyramid_attention._query_length = query_length;
    pyramid_attention._full_context_length = full_context_length;
    pyramid_attention._models = pyramid_models;
    pyramid_attention._attentions = pyramid_attentions;

    // Early return with pyramid attention result
    LOG_INFO("Returning pyramid attention with " << pyramid_models.size() << " models");
    LOG_INFO("  Query length: " << pyramid_attention._query_length);
    LOG_INFO("  Full context length: " << pyramid_attention._full_context_length);
    LOG_INFO("  Attention instances: " << pyramid_attention._attentions.size());
    return pyramid_attention;
}

}  // namespace function

namespace compiled {

// Constructor implementation
PyramidAttention::PyramidAttention(const function::PyramidAttention& func_pyramid)
    : query_size(func_pyramid._query_length),
      full_context_size(func_pyramid._full_context_length) {
    NPUW_ASSERT(func_pyramid._models.size() == func_pyramid._attentions.size());

    const size_t num_models = func_pyramid._models.size();
    _attention_infos.reserve(num_models);
    _context_lengths.reserve(num_models);
    _models.reserve(num_models);

    // Memory tracking setup
    const bool enable_memory_tracking = true;  // Could be a config option
    size_t initial_memory_kb = 0;

    if (enable_memory_tracking) {
        initial_memory_kb = get_process_memory_kb();
        LOG_INFO("=== PyramidAttention Memory Tracking Start: " << initial_memory_kb << " KB RSS ===");
    }

    // Process each model
    for (size_t i = 0; i < num_models; ++i) {
        size_t before_kb = enable_memory_tracking ? get_process_memory_kb() : 0;

        const auto& func_attn = func_pyramid._attentions[i];
        const auto& model = func_pyramid._models[i];

        // Build attention info
        PyramidAttentionInfo attention_info;
        attention_info.params.reserve(func_attn._inputs.size());

        for (const auto& input : func_attn._inputs) {
            std::size_t p_idx = model->get_parameter_index(input.param);
            attention_info.params.push_back({p_idx, input.dim});
        }

        attention_info.mask_idx = model->get_parameter_index(func_attn._mask);
        attention_info.query_size = func_attn.query_len();
        attention_info.context_length = func_attn.context_len();

        _attention_infos.push_back(std::move(attention_info));
        _context_lengths.push_back(attention_info.context_length);
        _models.push_back(model);

        if (enable_memory_tracking) {
            size_t after_kb = get_process_memory_kb();
            size_t increase_kb = (after_kb > before_kb) ? (after_kb - before_kb) : 0;
            LOG_DEBUG("Model[" << i << "]: RSS increased by " << increase_kb << " KB (" << (increase_kb / 1024)
                               << " MB), total: " << after_kb << " KB");
        }
    }

    if (enable_memory_tracking) {
        size_t final_memory_kb = get_process_memory_kb();
        size_t total_increase_kb = (final_memory_kb > initial_memory_kb) ? (final_memory_kb - initial_memory_kb) : 0;
        LOG_INFO("=== PyramidAttention Memory Tracking End: Total increase = "
                 << total_increase_kb << " KB (" << (total_increase_kb / 1024) << " MB) ===");
    }
}

}  // namespace compiled

namespace runtime {
namespace pyramid_attention {

// Pyramid Attention PositionIDs implementation
PositionIDs::PositionIDs(std::size_t param_idx, const compiled::PyramidAttention& d, const ov::ISyncInferRequest& rq)
    : m_position_ids_idx(param_idx),
      m_query_size(d.query_size),
      m_pyramid_attention(&d),
      m_rq(rq) {
    // FIXME: speculative decode is indistinguishable at this point!
    m_case = m_query_size == 1 ? Case::GENERATE : Case::PREFILL;
}

Selector::Ptr PositionIDs::find(const compiled::PyramidAttention& d, const ov::ISyncInferRequest& rq) {
    auto is_position_ids = [](const ov::Output<const ov::Node>& p) {
        const auto& shape = p.get_shape();
        // FIXME: 2D/3D position IDs are not supported here YET
        return p.get_node()->get_friendly_name() == "position_ids" &&
               (shape.size() == 1 || (shape.size() == 2 && shape[0] == 1));
    };

    const auto& inputs = rq.get_inputs();
    auto pos_ids_iter = std::find_if(inputs.begin(), inputs.end(), is_position_ids);
    if (pos_ids_iter != inputs.end()) {
        const auto param_idx = std::distance(inputs.begin(), pos_ids_iter);
        return Selector::Ptr{new PositionIDs(param_idx, d, rq)};
    }
    return Selector::Ptr{};
}

void PositionIDs::prepare(int64_t past_len) {
    const auto& iport = m_rq.get_compiled_model()->inputs()[m_position_ids_idx];
    const auto in_tensor = m_rq.get_tensor(iport);
    const auto in_dims = in_tensor->get_shape();

    // Same logic as regular attention PositionIDs
    auto* pos_data_ptr = in_tensor->data<int64_t>();
    for (auto idx = in_dims.back() - 1; idx >= 0; idx--) {
        if (pos_data_ptr[idx] > 0) {
            // Initialize fields
            m_current_length = pos_data_ptr[idx];
            switch (m_case) {
            case Case::GENERATE:
                // decode case, we have pos_id-1 past elements to take from kvcache
                m_past_length = m_current_length;
                break;
            case Case::PREFILL:
                // chunked prefill case. calculate the past_length in full chunks
                // FIXME: We know too much about chunking here
                m_past_length = ((past_len + m_query_size - 1) / m_query_size) * m_query_size;
                break;
            default:
                NPUW_ASSERT(false && "Reached the unreachable code");
            }

            // Select the optimal pyramid model based on current sequence length
            NPUW_ASSERT(m_pyramid_attention && "PyramidAttention reference must not be null");

            const auto& context_lengths = m_pyramid_attention->_context_lengths;
            const int64_t current_seq_length = m_query_size + m_past_length;

            // Find the smallest pyramid model that can handle the current sequence length
            for (std::size_t i = 0; i < context_lengths.size(); ++i) {
                if (current_seq_length <= static_cast<int64_t>(context_lengths[i])) {
                    m_pyramid_id = i;
                    return;
                }
            }

            // If sequence length exceeds all models' capacity, use the largest model
            m_pyramid_id = context_lengths.size() - 1;
            return;
        }
    }
    LOG_WARN("Dynamic selector - no data found in the feature?");
    m_current_length = -1;

    NPUW_ASSERT(m_pyramid_attention && "PyramidAttention reference must not be null");
    // Default to largest model if no data found (safest choice for unknown sequence length)
    m_pyramid_id = m_pyramid_attention->_context_lengths.size() - 1;
}

int64_t PositionIDs::length() const {
    return m_current_length;
}

int64_t PositionIDs::past_length() const {
    return m_past_length;
}

}  // namespace pyramid_attention
}  // namespace runtime
}  // namespace npuw
}  // namespace ov