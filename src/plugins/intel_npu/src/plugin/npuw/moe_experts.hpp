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
    size_t input_batch_size = 0;                            // Input batch size
    std::shared_ptr<ov::op::v0::Tile> tile_node = nullptr;  // The Tile operation node
    std::shared_ptr<ov::Node> input_node = nullptr;         // Input to Tile operation

    // Validation helper
    bool is_valid() const {
        return num_experts > 0 && expert_hidden_dim > 0 && tile_node != nullptr;
    }
};

// Structure to hold MoE downstream processing information (ReduceSum + QKV + attention postproc + downstream layers)
struct MoEDownstream {
    size_t total_experts_num = 0;                         // Total number of experts
    size_t active_experts_num = 0;                        // Number of active experts (K in top-K)
    size_t param_index = 0;                               // Index of the parameter with expert outputs
    std::shared_ptr<ov::Model> modified_model = nullptr;  // Model with modified input shape

    bool is_valid() const {
        return total_experts_num > 0 && active_experts_num > 0 && active_experts_num <= total_experts_num &&
               modified_model != nullptr;
    }
};

// Helper function to validate MoE expert model and extract necessary information
std::optional<MoEValidationResult> validate_and_setup_moe_expert(const std::shared_ptr<ov::Model>& model);

// Helper function to transform MoE expert model from batched to single expert
std::shared_ptr<ov::Model> transform_to_single_expert(const std::shared_ptr<ov::Model>& original_model,
                                                      const MoEValidationResult& validation_result);

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
    size_t _input_batch_size = 0;   // Input batch size

    // The transformed single-expert model
    std::shared_ptr<ov::Model> _single_expert_model = nullptr;

    // Original batched model (for reference)
    std::shared_ptr<ov::Model> _original_model = nullptr;

    // Tile operation information
    std::shared_ptr<ov::op::v0::Tile> _tile_op = nullptr;
    ov::Shape _original_tile_output_shape;    // Shape before transformation
    ov::Shape _single_expert_shape;           // Shape after transformation
    std::optional<size_t> _router_param_idx;  // Parameter index for router output

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
        return _num_experts > 0 && _expert_hidden_dim > 0 && _single_expert_model != nullptr;
    }

    size_t num_experts() const {
        return _num_experts;
    }

    size_t expert_hidden_dim() const {
        return _expert_hidden_dim;
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
        std::cout << "  Number of experts: " << _num_experts << std::endl;
        std::cout << "  Expert hidden dimension: " << _expert_hidden_dim << std::endl;
        std::cout << "  Input batch size: " << _input_batch_size << std::endl;
        std::cout << "  Original tile output shape: " << _original_tile_output_shape << std::endl;
        std::cout << "  Single expert shape: " << _single_expert_shape << std::endl;
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
    static std::optional<MoEExperts> from(const std::shared_ptr<ov::Model>& model, size_t active_experts_num = 0);
};

// Factory method to create MoEDownstream from a model (for downstream pattern)
std::optional<MoEDownstream> create_moe_downstream(const std::shared_ptr<ov::Model>& model, size_t active_experts_num);

}  // namespace function

namespace compiled {

// Compile-time MoE expert information
struct MoEExperts {
    size_t num_experts = 0;
    size_t expert_hidden_dim = 0;

    // Compiled single expert model
    ov::SoPtr<ov::ICompiledModel> _compiled_model;

    // Store model temporarily for compilation
    std::shared_ptr<ov::Model> _model_to_compile;

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
    size_t param_index = 0;

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
