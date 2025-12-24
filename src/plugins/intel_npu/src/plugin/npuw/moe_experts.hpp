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
    std::shared_ptr<ov::Node> input_node = nullptr;         // Input to Tile operation
    std::optional<size_t> router_param_idx;  // Parameter index for router output (from Multiply in output path)

    // Stage detection (based on output shape analysis)
    bool is_decoding_stage = false;      // True if detected as decoding (token_count == 1)
    size_t detected_active_experts = 1;  // Detected number of active experts (1 for prefill, K for decoding)
    bool has_reduce_sum = false;         // True if ReduceSum is present in output path (decoding stage)

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
// For prefill: num_target_experts = 1 (SINGLE_EXPERT mode)
// For decoding: num_target_experts = K (ACTIVE_EXPERTS mode)
std::shared_ptr<ov::Model> transform_moe_experts(const std::shared_ptr<ov::Model>& original_model,
                                                 MoEValidationResult& validation_result,
                                                 size_t num_target_experts = 1,
                                                 ExpertMode mode = ExpertMode::SINGLE_EXPERT);

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
    size_t _input_token_count = 0;  // Number of input tokens

    // Transformation mode and target expert count
    ExpertMode _mode = ExpertMode::SINGLE_EXPERT;  // Transformation mode
    size_t _num_active_experts = 1;  // Number of active experts (1 for SINGLE_EXPERT, K for ACTIVE_EXPERTS)

    // The transformed expert model (single expert or K active experts)
    std::shared_ptr<ov::Model> _single_expert_model = nullptr;

    // Original batched model (for reference)
    std::shared_ptr<ov::Model> _original_model = nullptr;

    // Tile operation information
    std::shared_ptr<ov::op::v0::Tile> _tile_op = nullptr;
    ov::Shape _original_tile_output_shape;    // Shape before transformation
    ov::Shape _single_expert_shape;           // Shape after transformation (single or K experts)
    std::optional<size_t> _router_param_idx;  // Parameter index for router output
    bool _has_reduce_sum = false;             // Whether ReduceSum is included (decoding stage)

    // Input/output information for the expert subgraph
    struct ExpertIO {
        std::string name;
        ov::element::Type element_type;
        ov::PartialShape shape;
    };
    std::vector<ExpertIO> _inputs;
    std::vector<ExpertIO> _outputs;

    // Validation helpers
    bool is_valid() const {
        return _num_experts > 0 && _expert_hidden_dim > 0 && _single_expert_model != nullptr &&
               _router_param_idx.has_value();
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

    bool has_reduce_sum() const {
        return _has_reduce_sum;
    }

    const std::shared_ptr<ov::Model>& single_expert_model() const {
        return _single_expert_model;
    }

    const std::shared_ptr<ov::Model>& original_model() const {
        return _original_model;
    }

    std::optional<size_t> router_param_idx() const {
        return _router_param_idx;
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
        std::cout << "  Has ReduceSum: " << (_has_reduce_sum ? "Yes" : "No") << std::endl;
        std::cout << "  Original tile output shape: " << _original_tile_output_shape << std::endl;
        std::cout << "  Transformed expert shape: " << _single_expert_shape << std::endl;
        std::cout << "  Inputs: " << _inputs.size() << std::endl;
        for (const auto& input : _inputs) {
            std::cout << "    - " << input.name << " [" << input.element_type << ", " << input.shape << "]"
                      << std::endl;
        }
        std::cout << "  Outputs: " << _outputs.size() << std::endl;
        for (const auto& output : _outputs) {
            std::cout << "    - " << output.name << " [" << output.element_type << ", " << output.shape << "]"
                      << std::endl;
        }
    }

    // Factory method to create MoEExperts from a model (for expert pattern only)
    // router_model: Router model to extract actual K from TopK node (required)
    static std::optional<MoEExperts> from(const std::shared_ptr<ov::Model>& model,
                                          const std::shared_ptr<ov::Model>& router_model);
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
    size_t input_token_count = 0;   // Number of input tokens (1 for decoding, >1 for prefill)
    function::ExpertMode mode = function::ExpertMode::SINGLE_EXPERT;
    bool has_reduce_sum = false;  // Whether ReduceSum is included

    // Compiled expert model (single expert or K active experts)
    ov::SoPtr<ov::ICompiledModel> _compiled_model;

    // Store model temporarily for compilation
    std::shared_ptr<ov::Model> _model_to_compile;

    // Router parameter index (from Multiply in output path)
    std::optional<size_t> _router_param_idx;

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
