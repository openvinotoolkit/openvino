// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <optional>
#include <vector>

#include "logging.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace npuw {

namespace function {

// Complete structure analysis result (immutable after analysis)
struct MoEStructureInfo {
    // Expert configuration
    size_t num_experts = 0;        // Total number of experts in the model
    size_t expert_hidden_dim = 0;  // Hidden dimension of single expert
    size_t input_token_count = 0;  // Number of input tokens

    // Key nodes in the graph
    std::shared_ptr<ov::op::v0::Tile> expert_input_tile_node = nullptr;  // Tile node for replicating expert inputs
    std::shared_ptr<ov::op::v1::Multiply> router_scores_multiply_node =
        nullptr;  // Multiply node in output path (expert_output * router_scores)

    // Parameter indices (detected once during analysis)
    std::optional<size_t> expert_input_param_idx;  // Parameter index for expert's input (token embeddings)
    std::optional<size_t> router_scores_idx;       // Parameter index for router scores (from Multiply in output path)

    // Stage inference
    bool is_decoding_stage = false;  // True if decoding (token_count == 1), false if prefill (token_count > 1)

    bool is_valid() const {
        return num_experts > 0 && expert_hidden_dim > 0 && expert_input_tile_node != nullptr &&
               router_scores_multiply_node != nullptr && expert_input_param_idx.has_value() &&
               router_scores_idx.has_value();
    }
};

// Expert transformation mode
enum class ExpertMode {
    SINGLE_EXPERT,  // Transform to 1 expert (prefill stage)
    ACTIVE_EXPERTS  // Transform to K active experts (decoding stage)
};

// Transformation configuration (clear decision parameters)
struct MoETransformConfig {
    ExpertMode mode;            // SINGLE_EXPERT for prefill, ACTIVE_EXPERTS for decoding
    size_t num_target_experts;  // Number of experts in transformed model (1 for prefill, K for decoding)
    size_t chunk_size;          // Token chunk size for prefill (0 for decoding)
};

// Structure to hold MoE downstream processing information (ReduceSum + QKV + attention postproc + downstream layers)
struct MoEDownstream {
    size_t total_experts_num = 0;                         // Total number of experts
    size_t active_experts_num = 0;                        // Number of active experts (K in top-K)
    size_t expert_output_param_idx = 0;                   // Index of the parameter that receives expert outputs
    std::shared_ptr<ov::Model> modified_model = nullptr;  // Model with modified input shape

    bool is_valid() const {
        return total_experts_num > 0 && active_experts_num > 0 && active_experts_num <= total_experts_num &&
               modified_model != nullptr;
    }
};

// Analyze MoE model structure and detect all necessary information
// This function is pure analysis - no modification to input model
std::optional<MoEStructureInfo> analyze_moe_structure(const std::shared_ptr<ov::Model>& model);

// Determine transformation parameters based on structure info and K from router
MoETransformConfig determine_transformation_params(const MoEStructureInfo& structure_info,
                                                   size_t k_from_router,
                                                   size_t prefill_chunk_size);

// Transform MoE expert model based on configuration (pure function, no side effects)
// Returns transformed model, does not modify any input parameters
std::shared_ptr<ov::Model> transform_moe_model(const std::shared_ptr<ov::Model>& original_model,
                                               const MoEStructureInfo& structure_info,
                                               const MoETransformConfig& config);

// Unroll the MoE expert model on the expert dimension using GraphRewrite patterns
// This creates separate computation branches for each expert
std::shared_ptr<ov::Model> unroll_expert_dimension(const std::shared_ptr<ov::Model>& model,
                                                   const MoEStructureInfo& structure_info,
                                                   size_t num_experts);

// Helper function to detect and transform MoE downstream pattern
// Looks for: Parameter -> Convert -> ReduceSum pattern
// If found, modifies input shape from [total_experts, 1, H, W] to [active_experts, 1, H, W]
std::optional<MoEDownstream> detect_and_transform_moe_downstream(const std::shared_ptr<ov::Model>& model,
                                                                 size_t active_experts_num);

// Structure to hold MoE expert information at partition-time
struct MoEExperts {
    // Basic information about the expert model
    size_t _num_experts = 0;        // Total number of experts in the model
    size_t _expert_hidden_dim = 0;  // Hidden dimension for a single expert
    size_t _input_token_count = 0;  // Number of input tokens (original total token count)
    size_t _chunk_token_count = 0;  // Chunk size for prefill mode (0 for decoding mode)

    // Transformation mode and target expert count
    ExpertMode _mode = ExpertMode::SINGLE_EXPERT;  // Transformation mode
    size_t _num_active_experts = 1;  // Number of active experts (1 for SINGLE_EXPERT, K for ACTIVE_EXPERTS)

    // Transformed expert models for different chunk sizes (chunk_size -> model)
    // For prefill: multiple models with different chunk sizes {32, 64, 128, 256, 512}
    // For decoding: single model with chunk_size = 0 (no chunking, K active experts)
    std::map<size_t, std::shared_ptr<ov::Model>> _transformed_models;

    // Parameter indices
    std::optional<size_t> _router_scores_idx;       // Parameter index for router scores
    std::optional<size_t> _expert_input_param_idx;  // Parameter index for expert's input (token embeddings)

    // Parameter mapping: original_param_idx -> [unrolled_param_indices]
    // For prefill (SINGLE_EXPERT): no unrolling, so this will be empty or identity mapping
    // For decoding (ACTIVE_EXPERTS): maps original params to K unrolled params
    // Same for all chunk sizes since unrolling only happens in decoding mode
    std::map<size_t, std::vector<size_t>> _param_mapping;

    // Validation helpers
    bool is_valid() const {
        return _num_experts > 0 && _expert_hidden_dim > 0 && !_transformed_models.empty() &&
               _router_scores_idx.has_value();
    }

    size_t num_experts() const {
        return _num_experts;
    }

    size_t expert_hidden_dim() const {
        return _expert_hidden_dim;
    }

    size_t num_active_experts() const {
        return _num_active_experts;
    }

    ExpertMode mode() const {
        return _mode;
    }

    const std::map<size_t, std::shared_ptr<ov::Model>>& transformed_models() const {
        return _transformed_models;
    }

    // Get transformed model for a specific chunk size
    std::shared_ptr<ov::Model> get_model_for_chunk_size(size_t chunk_size) const {
        auto it = _transformed_models.find(chunk_size);
        if (it != _transformed_models.end()) {
            return it->second;
        }
        return nullptr;
    }

    std::optional<size_t> router_scores_idx() const {
        return _router_scores_idx;
    }

    std::optional<size_t> expert_input_param_idx() const {
        return _expert_input_param_idx;
    }

    const std::map<size_t, std::vector<size_t>>& param_mapping() const {
        return _param_mapping;
    }

    // Log MoE expert information for debugging
    void log_info() const {
        std::cout << "MoE Expert Information:" << std::endl;
        std::cout << "  Mode: " << (_mode == ExpertMode::SINGLE_EXPERT ? "SINGLE_EXPERT" : "ACTIVE_EXPERTS")
                  << std::endl;
        std::cout << "  Number of experts: " << _num_experts << std::endl;
        std::cout << "  Number of active experts: " << _num_active_experts << std::endl;
        std::cout << "  Expert hidden dimension: " << _expert_hidden_dim << std::endl;
        std::cout << "  Input token count: " << _input_token_count << std::endl;

        if (_mode == ExpertMode::ACTIVE_EXPERTS) {
            std::cout << "  Decoding mode: no chunking" << std::endl;
        } else {
            std::cout << "  Prefill mode - Available chunk sizes: ";
            for (const auto& entry : _transformed_models) {
                std::cout << entry.first << " ";
            }
            std::cout << std::endl;
        }
    }

    // Factory method to create MoEExperts from a model (for expert pattern only)
    // router_model: Router model to extract actual K from TopK node (required)
    // chunk_size: Token chunk size for prefill processing (default: 128)
    static std::optional<MoEExperts> from(const std::shared_ptr<ov::Model>& model,
                                          const std::shared_ptr<ov::Model>& router_model,
                                          size_t prefill_chunk_size);
};

// Factory method to create MoEDownstream from a model (for downstream pattern)
// router_model: Router model to extract actual K from TopK node (required)
std::optional<MoEDownstream> create_moe_downstream(const std::shared_ptr<ov::Model>& model,
                                                   const std::shared_ptr<ov::Model>& router_model);

}  // namespace function

namespace compiled {

// Compile-time MoE expert information
struct MoEExperts {
    size_t num_experts = 0;
    size_t expert_hidden_dim = 0;
    size_t num_active_experts = 1;  // Number of active experts (1 for prefill, K for decoding)
    size_t input_token_count = 0;   // Number of input tokens (original total token count)
    function::ExpertMode mode = function::ExpertMode::SINGLE_EXPERT;

    // Compiled expert models for different chunk sizes (chunk_size -> compiled_model)
    std::map<size_t, ov::SoPtr<ov::ICompiledModel>> _compiled_models;

    // Store models temporarily for compilation (chunk_size -> model)
    std::map<size_t, std::shared_ptr<ov::Model>> _models_to_compile;

    // Router scores parameter index (from Multiply in output path)
    std::optional<size_t> _router_scores_idx;
    // Expert input parameter index (token embeddings)
    std::optional<size_t> _expert_input_param_idx;

    // Parameter mapping: original_param_idx -> [unrolled_param_indices]
    // Same for all chunk sizes (unrolling only in decoding mode)
    std::map<size_t, std::vector<size_t>> _param_mapping;

    MoEExperts() = default;

    // Constructor that extracts metadata and stores model for compilation
    explicit MoEExperts(const function::MoEExperts& func_moe);

    // Set compiled model for a specific chunk size after compilation completes
    void set_compiled_model(size_t chunk_size, ov::SoPtr<ov::ICompiledModel>&& compiled_model);

    // Get compiled model for a specific chunk size
    ov::SoPtr<ov::ICompiledModel> get_compiled_model(size_t chunk_size) const {
        auto it = _compiled_models.find(chunk_size);
        if (it != _compiled_models.end()) {
            return it->second;
        }
        return nullptr;
    }

    // Validation
    bool is_valid() const {
        return num_experts > 0 && expert_hidden_dim > 0 && !_compiled_models.empty();
    }
};

// Compiled MoE Downstream structure
struct MoEDownstream {
    size_t total_experts_num = 0;
    size_t active_experts_num = 0;
    size_t expert_output_param_idx = 0;  // Index of the parameter that receives expert outputs

    // Compiled modified downstream model
    ov::SoPtr<ov::ICompiledModel> _compiled_model;

    // Store model temporarily for compilation
    std::shared_ptr<ov::Model> _model_to_compile;

    MoEDownstream() = default;

    // Constructor that extracts metadata and stores model for compilation
    explicit MoEDownstream(const function::MoEDownstream& func_downstream);

    // Set compiled model after compilation completes
    void set_compiled_model(ov::SoPtr<ov::ICompiledModel>&& compiled_model);

    // Validation
    bool is_valid() const {
        return total_experts_num > 0 && active_experts_num > 0 && active_experts_num <= total_experts_num &&
               _compiled_model != nullptr;
    }
};

}  // namespace compiled

namespace runtime {
namespace moe_experts {

// TODO: Implement runtime dispatcher for MoE experts
// This will execute individual experts during inference

}  // namespace moe_experts
}  // namespace runtime

}  // namespace npuw
}  // namespace ov
