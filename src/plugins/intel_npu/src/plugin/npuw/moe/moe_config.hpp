// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <vector>

#include "moe_types.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {
namespace moe {

/**
 * @brief MoE compilation-time configuration (read-only)
 *
 * Contains static configuration extracted from CompiledModel::MoEExperts.
 * This structure is initialized once during prepare() and remains constant
 * throughout the inference lifecycle.
 */
struct MoEConfig {
    // Expert topology configuration
    size_t num_experts = 0;         // Total number of experts (e.g., 32)
    size_t num_active_experts = 0;  // Number of experts activated per token (e.g., 4)
    size_t input_token_count = 0;   // Number of input tokens
    size_t expert_hidden_dim = 0;   // Expert output embedding dimension

    // Parameter mapping for unrolled models
    // Maps original parameter index to unrolled parameter indices
    // Example: param[5] -> [param[10], param[11], param[12], param[13]] for K=4 experts
    std::map<size_t, std::vector<size_t>> param_mapping;

    // Input parameter indices
    std::optional<size_t> router_scores_idx;       // Index of router scores input
    std::optional<size_t> expert_input_param_idx;  // Index of expert input (token embeddings)

    // Compiled models for different chunk sizes (for EXPERT_ITERATIVE mode)
    // Key: chunk_size (e.g., 256, 128, 64, 32, 16)
    // Value: Compiled model for that chunk size
    std::map<size_t, ov::SoPtr<ov::ICompiledModel>> compiled_models;

    /**
     * @brief Get processing mode based on input token count
     * @return EXPERT_BATCH for single token, EXPERT_ITERATIVE for multiple tokens
     */
    MoEProcessingMode get_processing_mode() const {
        return determine_processing_mode(input_token_count);
    }

    /**
     * @brief Check if using expert batch mode
     * Single token processed with K experts in parallel
     */
    bool is_expert_batch_mode() const {
        return get_processing_mode() == MoEProcessingMode::EXPERT_BATCH;
    }

    /**
     * @brief Check if using expert iterative mode
     * Multiple tokens processed by iterating through experts
     */
    bool is_expert_iterative_mode() const {
        return get_processing_mode() == MoEProcessingMode::EXPERT_ITERATIVE;
    }

    /**
     * @brief Validate configuration consistency
     */
    bool is_valid() const {
        return num_experts > 0 && num_active_experts > 0 && num_active_experts <= num_experts &&
               input_token_count > 0 && expert_hidden_dim > 0 && !compiled_models.empty();
    }
};

}  // namespace moe
}  // namespace npuw
}  // namespace ov
