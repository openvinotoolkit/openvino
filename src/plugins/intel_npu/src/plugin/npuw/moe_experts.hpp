// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "logging.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace npuw {

namespace function {

// Helper struct to hold validation results for MoE expert model
struct MoEValidationResult {
    size_t num_experts = 0;                                 // Total number of experts
    size_t expert_hidden_dim = 0;                           // Hidden dimension of single expert
    size_t input_token_count = 0;                           // Number of input tokens
    std::shared_ptr<ov::op::v0::Tile> tile_node = nullptr;  // The Tile operation node
    std::optional<size_t> router_scores_idx;       // Parameter index for router scores (from Multiply in output path)
    std::optional<size_t> expert_input_param_idx;  // Parameter index for expert's input (token embeddings)

    // Stage detection (based on output shape analysis)
    bool is_decoding_stage = false;      // True if detected as decoding (token_count == 1)
    size_t detected_active_experts = 1;  // Detected number of active experts (1 for prefill, K for decoding)

    // Validation helper
    bool is_valid() const {
        return num_experts > 0 && expert_hidden_dim > 0 && tile_node != nullptr;
    }
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

// Helper function to validate MoE expert model and extract necessary information
std::optional<MoEValidationResult> validate_and_setup_moe_expert(const std::shared_ptr<ov::Model>& model,
                                                                 size_t active_experts_num);

// Expert transformation mode
enum class ExpertMode {
    SINGLE_EXPERT,  // Transform to 1 expert (prefill stage)
    ACTIVE_EXPERTS  // Transform to K active experts (decoding stage)
};

// Helper function to transform MoE expert model from batched to target number of experts
// For prefill: num_target_experts = 1 (SINGLE_EXPERT mode), token_count = chunk_size
// For decoding: num_target_experts = K (ACTIVE_EXPERTS mode), token_count = 1
std::shared_ptr<ov::Model> transform_moe_experts(const std::shared_ptr<ov::Model>& original_model,
                                                 MoEValidationResult& validation_result,
                                                 size_t num_target_experts,
                                                 ExpertMode mode,
                                                 size_t prefill_chunk_size = 0);

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

    // The transformed expert model (1 expert for prefill, K experts for decoding)
    std::shared_ptr<ov::Model> _transformed_model = nullptr;

    // Parameter indices
    std::optional<size_t> _router_scores_idx;       // Parameter index for router scores
    std::optional<size_t> _expert_input_param_idx;  // Parameter index for expert's input (token embeddings)

    // Validation helpers
    bool is_valid() const {
        return _num_experts > 0 && _expert_hidden_dim > 0 && _transformed_model != nullptr &&
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

    const std::shared_ptr<ov::Model>& transformed_model() const {
        return _transformed_model;
    }

    std::optional<size_t> router_scores_idx() const {
        return _router_scores_idx;
    }

    std::optional<size_t> expert_input_param_idx() const {
        return _expert_input_param_idx;
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
        std::cout << "  Chunk token count: " << _chunk_token_count << std::endl;
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
    size_t chunk_token_count = 0;   // Chunk size for prefill mode (0 for decoding mode)
    function::ExpertMode mode = function::ExpertMode::SINGLE_EXPERT;

    // Compiled expert model (single expert or K active experts)
    ov::SoPtr<ov::ICompiledModel> _compiled_model;

    // Store model temporarily for compilation
    std::shared_ptr<ov::Model> _model_to_compile;

    // Router scores parameter index (from Multiply in output path)
    std::optional<size_t> _router_scores_idx;
    // Expert input parameter index (token embeddings)
    std::optional<size_t> _expert_input_param_idx;

    MoEExperts() = default;

    // Constructor that extracts metadata and stores model for compilation
    explicit MoEExperts(const function::MoEExperts& func_moe);

    // Set compiled model after compilation completes
    void set_compiled_model(ov::SoPtr<ov::ICompiledModel>&& compiled_model);

    // Validation
    bool is_valid() const {
        return num_experts > 0 && expert_hidden_dim > 0 && _compiled_model != nullptr;
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
