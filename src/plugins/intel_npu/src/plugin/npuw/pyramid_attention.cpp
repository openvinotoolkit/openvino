// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyramid_attention.hpp"

#include <algorithm>
#include <utility>

#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/util/op_types.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {
namespace function {

// Helper function to create Attention instance from a model
std::optional<ov::npuw::function::Attention> create_attention_from_model(
    const std::shared_ptr<ov::Model>& model,
    const std::map<std::string, size_t>& past_key_sequence_dims,
    const std::map<std::string, size_t>& past_value_sequence_dims) {
    // Find SDPA pattern nodes in the model
    auto pattern_nodes = ov::npuw::util::find_sdpa_pattern_nodes(model);
    if (!pattern_nodes.is_valid()) {
        LOG_WARN("Could not find SDPA pattern in model");
        return std::nullopt;
    }

    // Find mask parameter in the model
    auto mask_param = ov::npuw::util::find_mask_parameter(pattern_nodes.add_node);
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
        if (ov::npuw::util::isPastKeyValuesKeyContiguous(param_name).has_value()) {
            auto dim_iter = past_key_sequence_dims.find(param_name);
            if (dim_iter != past_key_sequence_dims.end()) {
                attention._inputs.push_back(ov::npuw::function::Attention::Param{param, dim_iter->second});
            }
        } else if (ov::npuw::util::isPastKeyValuesValueContiguous(param_name).has_value()) {
            auto dim_iter = past_value_sequence_dims.find(param_name);
            if (dim_iter != past_value_sequence_dims.end()) {
                attention._inputs.push_back(ov::npuw::function::Attention::Param{param, dim_iter->second});
            }
        }
    }

    return attention;
}

// Collect past KV block parameter indices from a Concat node in Concat input order.
// All inputs except the last (present_key/value) are past block params.
// SplitKVCacheIntoBlocks may insert a Convert between each block Parameter and the Concat.
static void collect_concat_block_indices(const std::shared_ptr<ov::Model>& model,
                                         const std::shared_ptr<ov::Node>& concat_node,
                                         std::vector<size_t>& out) {
    if (!concat_node) {
        return;
    }
    const size_t n = concat_node->get_input_size();
    for (size_t i = 0; i + 1 < n; ++i) {
        auto node = concat_node->get_input_node_shared_ptr(i);
        if (auto cvt = std::dynamic_pointer_cast<ov::op::v0::Convert>(node)) {
            node = cvt->input_value(0).get_node_shared_ptr();
        }
        if (auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(node)) {
            out.push_back(static_cast<size_t>(model->get_parameter_index(param)));
        }
    }
}

// Helper function to process a single pyramid model (clone, reshape, patch, optimize)
std::optional<PyramidModelResult> process_pyramid_model(const std::shared_ptr<ov::Model>& original_model,
                                                        size_t model_idx,
                                                        size_t pyramid_step,
                                                        size_t query_length,
                                                        size_t full_past_kv_length,
                                                        size_t full_context_length,
                                                        const std::map<std::string, size_t>& past_key_sequence_dims,
                                                        const std::map<std::string, size_t>& past_value_sequence_dims,
                                                        bool is_block_split) {
    // Clone the original model for modification
    auto cloned_model = original_model->clone();

    // Calculate dimensions for this model
    size_t current_context_length = 0u;
    size_t current_past_length = 0u;

    // FIXME: SPECULATIVE CASE!!!
    if (query_length == 1u) {
        // GENERATE
        auto output_len = full_context_length - query_length - full_past_kv_length;
        current_context_length = (model_idx + 1) * pyramid_step + output_len;
        current_past_length = current_context_length - query_length;
    } else {
        // PREFILL
        current_context_length = (model_idx + 1) * pyramid_step;
        current_past_length = current_context_length - query_length;
    }
    // FIXME: Probably the generic formula for all cases is:
    // current_context_length = (model_idx + 1) * pyramid_step;
    // current_past = current_context_length - query_length
    // - should work for the speculative case as well
    LOG_DEBUG("Model " << model_idx << ":");
    LOG_DEBUG("  Context length: " << current_context_length);
    LOG_DEBUG("  Past length: " << current_past_length);

    // -------------------------------------------------------------------------
    // Block-split KV cache path (is_block_split == true)
    //
    // The model has already been through SplitKVCacheIntoBlocks. The KV Concat
    // now has the form:
    //   Concat([past_key_block_0, ..., past_key_block_{N-2}, present_key])
    // For pyramid model[idx] we only need `idx` past blocks, so we replace the
    // Concat with a smaller one and remove the surplus block Parameters.
    //   model[0]  -> Concat([present_key])           (0 past blocks)
    //   model[1]  -> Concat([block_0, present_key])  (1 past block)
    //   model[idx]-> Concat([block_0..block_{idx-1}, present_key])
    // -------------------------------------------------------------------------
    if (is_block_split) {
        const size_t num_blocks_needed = model_idx;  // model[idx] needs exactly idx past blocks

        // Lambda: shrink one Concat (key or value) to keep only num_blocks_needed past inputs.
        // Returns the new (shrunk) Concat node on success, nullptr on error.
        auto shrink_concat_inputs = [&](const std::shared_ptr<ov::Node>& concat_node) -> std::shared_ptr<ov::Node> {
            auto concat = std::dynamic_pointer_cast<ov::op::v0::Concat>(concat_node);
            if (!concat) {
                LOG_WARN("  Block shrink: expected a Concat node, got something else");
                return nullptr;
            }

            const size_t total_inputs = concat->get_input_size();
            // The last input is always present_key/value (SplitKVCacheIntoBlocks convention).
            const size_t num_global_past_blocks = total_inputs - 1u;

            if (num_blocks_needed > num_global_past_blocks) {
                LOG_WARN("  Block shrink: model[" << model_idx << "] needs " << num_blocks_needed
                                                  << " blocks but global model only has " << num_global_past_blocks);
                return nullptr;
            }

            ov::OutputVector new_inputs;
            new_inputs.reserve(num_blocks_needed + 1u);
            std::vector<std::shared_ptr<ov::op::v0::Parameter>> params_to_remove;

            for (size_t i = 0; i < num_global_past_blocks; ++i) {
                auto src_output = concat->input_value(i);
                auto src_node = src_output.get_node_shared_ptr();

                // SplitKVCacheIntoBlocks may insert a Convert node between the block
                // Parameter and the Concat; unwrap it to reach the Parameter.
                std::shared_ptr<ov::op::v0::Parameter> block_param;
                if (auto cvt = std::dynamic_pointer_cast<ov::op::v0::Convert>(src_node)) {
                    block_param =
                        std::dynamic_pointer_cast<ov::op::v0::Parameter>(cvt->input_value(0).get_node_shared_ptr());
                } else {
                    block_param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(src_node);
                }

                if (i < num_blocks_needed) {
                    new_inputs.push_back(src_output);  // keep
                } else {
                    if (block_param) {
                        params_to_remove.push_back(block_param);
                        LOG_DEBUG("  Dropping block param: " << block_param->get_friendly_name());
                    }
                }
            }
            // Always keep the present_key/value (last input).
            new_inputs.push_back(concat->input_value(total_inputs - 1u));

            auto new_concat = std::make_shared<ov::op::v0::Concat>(new_inputs, concat->get_axis());
            new_concat->set_friendly_name(concat->get_friendly_name());
            concat->output(0).replace(new_concat->output(0));

            for (const auto& param : params_to_remove) {
                cloned_model->remove_parameter(param);
            }

            LOG_DEBUG("  Concat shrunken: " << total_inputs << " inputs -> " << new_inputs.size() << " inputs");
            return new_concat;
        };

        // Find SDPA pattern in the cloned model.
        auto cloned_pattern = ov::npuw::util::find_sdpa_pattern_nodes(cloned_model);
        if (!cloned_pattern.is_valid()) {
            LOG_WARN("Could not find SDPA pattern in block-mode cloned model (model_idx=" << model_idx << ")");
            return std::nullopt;
        }

        // Capture value Concat axis before shrinking — needed for patch_reshape_constants below.
        int64_t value_concat_axis = 0;
        if (auto vc = std::dynamic_pointer_cast<ov::op::v0::Concat>(cloned_pattern.past_value_concat_node)) {
            const auto& out_shape = vc->get_output_partial_shape(0);
            value_concat_axis = ov::util::try_normalize_axis(vc->get_axis(), out_shape.rank(), *vc);
        }

        auto shrunk_key = shrink_concat_inputs(cloned_pattern.past_key_concat_node);
        auto shrunk_value = shrink_concat_inputs(cloned_pattern.past_value_concat_node);
        if (!shrunk_key || !shrunk_value) {
            LOG_WARN("Failed to shrink Concat nodes for block-mode pyramid model[" << model_idx << "]");
            return std::nullopt;
        }

        LOG_DEBUG("  Concat nodes shrunk successfully for model[" << model_idx << "]");

        // Apply the same pre-reshape patching + reshape sequence as the non-block path:
        //   1. patch_broadcast_constants — fix Broadcast shape constants referencing full_context_length
        //   2. patch_reshape_constants  — set seq dim to -1 in value-path Reshape shape constant
        //                                 (pattern: value_Concat → Reshape → MatMul2 ← Softmax)
        //   3. reshape(new_shapes)      — apply mask shape update and propagate all shapes
        //   4. validate_nodes_and_infer_types
        ov::npuw::function::patch_broadcast_constants(cloned_model, full_context_length);
        // The map key is unused by patch_reshape_constants; only the dim-index value matters.
        ov::npuw::function::patch_reshape_constants(cloned_model, {{"", static_cast<size_t>(value_concat_axis)}});

        // Directly set partial shape on the Parameter node — bypasses reshape() name/pointer
        // lookup entirely. reshape(Output<Node>) can silently miss, reshape(string) uses
        // tensor names (not friendly names), so set_partial_shape() is the safest path.
        auto mask_param = ov::npuw::util::find_mask_parameter(cloned_pattern.add_node);
        if (!mask_param) {
            LOG_WARN("Could not find mask parameter for block-mode pyramid model[" << model_idx << "]");
            return std::nullopt;
        }
        ov::PartialShape new_mask_shape = mask_param->get_partial_shape();
        if (new_mask_shape.rank().is_static() && new_mask_shape.rank().get_length() > 0) {
            new_mask_shape[new_mask_shape.rank().get_length() - 1] = current_context_length;
            mask_param->set_partial_shape(new_mask_shape);
            LOG_INFO("  Set mask '" << mask_param->get_friendly_name() << "' partial shape -> " << new_mask_shape);
        }

        cloned_model->validate_nodes_and_infer_types();
        ov::npuw::function::Attention block_attention;
        block_attention._mask = mask_param;
        block_attention._mask_shape = mask_param->get_shape();
        collect_concat_block_indices(cloned_model, shrunk_key, block_attention.past_key_block_variant_param_indices);
        collect_concat_block_indices(cloned_model,
                                     shrunk_value,
                                     block_attention.past_value_block_variant_param_indices);

        LOG_INFO("Block-mode pyramid model[" << model_idx << "] ready: " << num_blocks_needed
                                             << " past block(s), context=" << current_context_length);

        return PyramidModelResult{cloned_model, std::move(block_attention)};
    }
    // -------------------------------------------------------------------------
    // End block-split path
    // -------------------------------------------------------------------------

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

        if (param == cloned_mask_param) {
            // Handle attention mask parameter - use the mask parameter found in cloned model
            // Update the last dimension to current context length
            if (new_shape.size() >= 1) {
                new_shape[new_shape.size() - 1] = current_context_length;
                new_shapes[param->output(0)] = new_shape;
                LOG_DEBUG("  Mask param '" << param_name << "' shape: " << original_shape << " -> " << new_shape);
            }
        } else if (ov::npuw::util::isPastKeyValuesKeyContiguous(param_name).has_value()) {
            // Handle past key parameters
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
        } else if (ov::npuw::util::isPastKeyValuesValueContiguous(param_name).has_value()) {
            // Handle past value parameters
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
    ov::npuw::function::patch_broadcast_constants(cloned_model, full_context_length);
    ov::npuw::function::patch_reshape_constants(cloned_model, past_value_sequence_dims);

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
    auto pattern_nodes = ov::npuw::util::find_sdpa_pattern_nodes(model);
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

    // Detect block-split KV cache layout.
    // After SplitKVCacheIntoBlocks the single past_key param becomes N block params named
    // "{original_name}_block_{i}". We detect this by either count > 1 or the name suffix.
    const bool is_block_split =
        !pattern_nodes.past_key_param_nodes.empty() &&
        (pattern_nodes.past_key_param_nodes.size() > 1 ||
         !ov::npuw::util::isPastKeyValuesKeyContiguous(pattern_nodes.past_key_param_nodes[0]->get_friendly_name())
              .has_value());

    if (is_block_split) {
        LOG_INFO("Detected block-split KV cache (" << pattern_nodes.past_key_param_nodes.size()
                                                   << " past key block(s)): using Concat-shrink path");
        // Compute full-model block param indices from the **Concat input order** (block_0, block_1, ..., present).
        // This is the same order used by process_pyramid_model / collect_concat_block_indices, so
        // global and variant indices are always aligned for bind_block_ports.
        // Do NOT use past_key_param_nodes order (model parameter list) — it may differ.
        std::vector<size_t> full_key_indices, full_val_indices;
        collect_concat_block_indices(model, pattern_nodes.past_key_concat_node, full_key_indices);
        collect_concat_block_indices(model, pattern_nodes.past_value_concat_node, full_val_indices);
        return PyramidValidationBlockResult{query_length,
                                            full_context_length,
                                            std::move(full_key_indices),
                                            std::move(full_val_indices)};
    }

    // Pre-analyze original model to find sequence dimensions for past key/value parameters
    // This avoids repeated analysis in each cloned model
    size_t past_kv_length = 0;
    std::map<std::string, size_t> past_key_sequence_dims;
    std::map<std::string, size_t> past_value_sequence_dims;

    // Helper lambda to extract sequence dimensions from Concat node and assign to parameters
    auto extract_sequence_dims = [&](const std::shared_ptr<ov::Node>& concat_node,
                                     bool is_key,
                                     std::map<std::string, size_t>& sequence_dims) -> bool {
        if (!concat_node) {
            LOG_WARN("Could not find Concat node for past " << (is_key ? "key" : "value") << " in SDPA pattern");
            return false;
        }

        auto concat_op = std::dynamic_pointer_cast<ov::op::v0::Concat>(concat_node);
        NPUW_ASSERT(concat_op != nullptr);
        const auto& concat_out_shape = concat_op->get_output_partial_shape(0);
        const auto concat_axis =
            ov::util::try_normalize_axis(concat_op->get_axis(), concat_out_shape.rank(), *concat_op);

        LOG_DEBUG("Found past " << (is_key ? "key" : "value") << " Concat node, concat axis: " << concat_axis);

        // Find all matching parameters and assign this dimension
        const auto& original_params = model->get_parameters();
        for (const auto& param : original_params) {
            const std::string param_name = param->get_friendly_name();
            bool is_target_param = is_key ? ov::npuw::util::isPastKeyValuesKeyContiguous(param_name).has_value()
                                          : ov::npuw::util::isPastKeyValuesValueContiguous(param_name).has_value();

            if (is_target_param) {
                sequence_dims[param_name] = concat_axis;
                LOG_DEBUG("Assigned " << (is_key ? "key" : "value") << " concat axis " << concat_axis
                                      << " to parameter: " << param_name);

                auto curr_past_kv_length = param->get_shape()[concat_axis];
                if (past_kv_length && past_kv_length != curr_past_kv_length) {
                    LOG_WARN("Inconsistent past KV lengths found among " << (is_key ? "key" : "value")
                                                                         << " parameters");
                    return false;
                }
                past_kv_length = curr_past_kv_length;
            }
        }
        return true;
    };

    // Extract sequence dimensions for past key and past value
    if (!extract_sequence_dims(pattern_nodes.past_key_concat_node, true, past_key_sequence_dims) ||
        !extract_sequence_dims(pattern_nodes.past_value_concat_node, false, past_value_sequence_dims)) {
        return std::nullopt;
    }

    if (past_key_sequence_dims.empty() || past_value_sequence_dims.empty()) {
        LOG_WARN("Failed to find past KV parameters");
        return std::nullopt;
    }

    return PyramidValidationContiguousResult{query_length,
                                             full_context_length,
                                             past_kv_length,
                                             std::move(past_key_sequence_dims),
                                             std::move(past_value_sequence_dims)};
}

std::optional<PyramidAttention> PyramidAttention::from(const std::shared_ptr<ov::Model>& model) {
    // Validate and setup pyramid attention
    auto validation_result = validate_and_setup_pyramid_attention(model);
    if (!validation_result) {
        return std::nullopt;
    }

    // Extract shared fields and mode-specific fields from the variant.
    size_t query_length = 0;
    size_t full_past_kv_length = 0;
    size_t full_context_length = 0;
    std::map<std::string, size_t> past_key_sequence_dims;
    std::map<std::string, size_t> past_value_sequence_dims;
    bool is_block_split = false;
    std::vector<size_t> block_key_global_indices;
    std::vector<size_t> block_val_global_indices;

    std::visit(
        [&](auto&& result) {
            using T = std::decay_t<decltype(result)>;
            query_length = result.query_length;
            full_context_length = result.full_context_length;
            if constexpr (std::is_same_v<T, PyramidValidationContiguousResult>) {
                full_past_kv_length = result.past_kv_length;
                past_key_sequence_dims = result.past_key_sequence_dims;
                past_value_sequence_dims = result.past_value_sequence_dims;
            } else {
                static_assert(std::is_same_v<T, PyramidValidationBlockResult>);
                is_block_split = true;
                block_key_global_indices = result.past_key_block_global_param_indices;
                block_val_global_indices = result.past_value_block_global_param_indices;
            }
        },
        *validation_result);

    std::vector<std::shared_ptr<ov::Model>> pyramid_models;

    // Use step 1024 to generate attention pyramid if it is the GENERATE case.
    // FIXME: Make it configurable
    // FIXME: Handle the speculative case here (query_length > 1; << 1024)
    bool is_generate = query_length == 1;
    size_t pyramid_step = is_generate ? 1024u : query_length;
    // FIXME: Check all the right alignments
    size_t num_models = full_context_length / pyramid_step;
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

            // In block mode the last model IS the full/original model (N blocks).
            // Propagate the full-model block indices into this variant's attention so
            // The info structs (PyramidAttentionContiguousInfo / PyramidAttentionBlockInfo) can copy them without
            // re-scanning the graph.
            if (is_block_split) {
                last_attention->past_key_block_variant_param_indices = block_key_global_indices;
                last_attention->past_value_block_variant_param_indices = block_val_global_indices;
            }

            pyramid_attentions.push_back(std::move(*last_attention));
            LOG_INFO("Successfully setup attention for original model[" << model_idx << "]");
        } else {
            // Process pyramid models 0 to num_models-2 using the helper function
            auto result = process_pyramid_model(model,
                                                model_idx,
                                                pyramid_step,
                                                query_length,
                                                full_past_kv_length,
                                                full_context_length,
                                                past_key_sequence_dims,
                                                past_value_sequence_dims,
                                                is_block_split);
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
    // Block indices are empty in contiguous mode; assigned unconditionally for simplicity.
    pyramid_attention.past_key_block_global_param_indices = std::move(block_key_global_indices);
    pyramid_attention.past_value_block_global_param_indices = std::move(block_val_global_indices);

    LOG_INFO("Returning pyramid attention with " << pyramid_models.size() << " models");
    LOG_INFO("  Query length: " << pyramid_attention._query_length);
    LOG_INFO("  Full context length: " << pyramid_attention._full_context_length);
    LOG_INFO("  Attention instances: " << pyramid_attention._attentions.size());
    return pyramid_attention;
}

}  // namespace function

namespace compiled {

// ── Static empty containers (returned by contiguous subclass block-mode stubs) ────
static const std::unordered_set<size_t> s_empty_port_set;
static const std::unordered_map<size_t, size_t> s_empty_port_map;

// ── PyramidAttentionContiguous virtual implementations ────────────────────────────

const std::unordered_set<size_t>& PyramidAttentionContiguous::key_block_port_set_at(size_t) const {
    return s_empty_port_set;
}
const std::unordered_set<size_t>& PyramidAttentionContiguous::val_block_port_set_at(size_t) const {
    return s_empty_port_set;
}
const std::unordered_map<size_t, size_t>& PyramidAttentionContiguous::key_block_port_map_at(size_t) const {
    return s_empty_port_map;
}
const std::unordered_map<size_t, size_t>& PyramidAttentionContiguous::val_block_port_map_at(size_t) const {
    return s_empty_port_map;
}

std::optional<std::size_t> PyramidAttentionContiguous::kv_param_dim(size_t pyramid_id, size_t input_idx) const {
    const auto& params = _attention_infos[pyramid_id].params;
    auto it = std::find_if(params.begin(), params.end(), [input_idx](const auto& p) {
        return p.idx == input_idx;
    });
    if (it == params.end()) {
        return std::nullopt;
    }
    return it->dim;
}

void PyramidAttentionContiguous::collect_strided_input_names(const ov::Model& model, std::string& out) const {
    if (_attention_infos.empty()) {
        return;
    }
    for (const auto& param : _attention_infos[0].params) {
        if (!out.empty()) {
            out += ",";
        }
        out += model.inputs()[param.idx].get_any_name();
    }
}

// ── PyramidAttention::make() static factory ───────────────────────────────────────

std::shared_ptr<PyramidAttention> PyramidAttention::make(const function::PyramidAttention& func_pyramid) {
    NPUW_ASSERT(func_pyramid._models.size() == func_pyramid._attentions.size());

    const size_t num_models = func_pyramid._models.size();
    const auto& gk = func_pyramid.past_key_block_global_param_indices;
    const auto& gv = func_pyramid.past_value_block_global_param_indices;
    const bool is_block = !gk.empty();

    LOG_INFO("Constructing compiled::PyramidAttention (" << (is_block ? "block" : "contiguous") << ") with "
                                                         << num_models << " models");

    if (!is_block) {
        auto obj = std::make_shared<PyramidAttentionContiguous>();
        obj->query_size = func_pyramid._query_length;
        obj->full_context_size = func_pyramid._full_context_length;
        obj->_models_to_compile = func_pyramid._models;
        obj->_attention_infos.reserve(num_models);
        obj->_context_lengths.reserve(num_models);

        for (size_t i = 0; i < num_models; ++i) {
            const auto& func_attn = func_pyramid._attentions[i];
            const auto& model = func_pyramid._models[i];
            PyramidAttentionContiguousInfo info;
            info.params.reserve(func_attn._inputs.size());
            for (const auto& input : func_attn._inputs) {
                info.params.push_back({static_cast<std::size_t>(model->get_parameter_index(input.param)), input.dim});
            }
            info.mask_idx = static_cast<std::size_t>(model->get_parameter_index(func_attn._mask));
            info.query_size = func_attn.query_len();
            info.context_length = func_attn.context_len();
            obj->_context_lengths.push_back(info.context_length);
            obj->_attention_infos.push_back(std::move(info));
        }
        LOG_INFO("compiled::PyramidAttentionContiguous metadata extracted");
        return obj;
    } else {
        auto obj = std::make_shared<PyramidAttentionBlock>();
        obj->query_size = func_pyramid._query_length;
        obj->full_context_size = func_pyramid._full_context_length;
        obj->_models_to_compile = func_pyramid._models;
        obj->past_key_block_global_param_indices = gk;
        obj->past_value_block_global_param_indices = gv;
        obj->_attention_infos.reserve(num_models);
        obj->_context_lengths.reserve(num_models);

        constexpr size_t NO_PORT = std::numeric_limits<size_t>::max();
        for (size_t i = 0; i < num_models; ++i) {
            const auto& func_attn = func_pyramid._attentions[i];
            const auto& model = func_pyramid._models[i];
            PyramidAttentionBlockInfo info;
            info.mask_idx = static_cast<std::size_t>(model->get_parameter_index(func_attn._mask));
            info.query_size = func_attn.query_len();
            info.context_length = func_attn.context_len();

            const auto& vk = func_attn.past_key_block_variant_param_indices;
            const auto& vv = func_attn.past_value_block_variant_param_indices;
            for (size_t m = 0; m < gk.size(); ++m) {
                const size_t port = (m < vk.size()) ? vk[m] : NO_PORT;
                info.past_key_block_port_map[gk[m]] = port;
                if (port != NO_PORT) {
                    info.past_key_block_port_set.insert(port);
                }
            }
            for (size_t m = 0; m < gv.size(); ++m) {
                const size_t port = (m < vv.size()) ? vv[m] : NO_PORT;
                info.past_value_block_port_map[gv[m]] = port;
                if (port != NO_PORT) {
                    info.past_value_block_port_set.insert(port);
                }
            }
            obj->_context_lengths.push_back(info.context_length);
            obj->_attention_infos.push_back(std::move(info));
        }
        LOG_INFO("compiled::PyramidAttentionBlock metadata extracted, " << gk.size() << " K blocks / " << gv.size()
                                                                        << " V blocks in full model");
        return obj;
    }
}

// Set compiled models after parallel compilation and clear temporary storage.
// Block indices are already populated in the constructor from the graph.
void PyramidAttention::set_compiled_models(std::vector<ov::SoPtr<ov::ICompiledModel>>&& compiled_models) {
    NPUW_ASSERT(compiled_models.size() == num_models() && "Compiled models count must match metadata count");
    _compiled_models = std::move(compiled_models);

    // Clear temporary models storage
    _models_to_compile.clear();
    _models_to_compile.shrink_to_fit();

    LOG_INFO("compiled::PyramidAttention compiled models set (" << _compiled_models.size() << " models)");
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
    for (int64_t idx = static_cast<int64_t>(in_dims.back()) - 1; idx >= 0; idx--) {
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
