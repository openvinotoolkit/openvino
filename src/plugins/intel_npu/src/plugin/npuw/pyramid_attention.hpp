// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "attention.hpp"

namespace ov {
namespace npuw {
namespace function {

// Helper struct to hold validation and setup results
struct PyramidValidationResult {
    size_t query_length;
    size_t full_context_length;
    std::map<std::string, size_t> past_key_sequence_dims;
    std::map<std::string, size_t> past_value_sequence_dims;
};

// Helper struct to hold model processing result
struct PyramidModelResult {
    std::shared_ptr<ov::Model> model;
    ov::npuw::function::Attention attention;
};

// Helper function to patch broadcast constants (set to 1 for dynamic handling)
void patch_broadcast_constants(const std::shared_ptr<ov::Model>& model, size_t target_length);

// Helper function to patch reshape constants for pre-reshape (-1 substitution)
void patch_reshape_constants_pre_reshape(const std::shared_ptr<ov::Model>& model,
                                         const ov::npuw::function::Attention& dyn);

// Helper function to process a single pyramid model (clone, reshape, patch, optimize)
std::optional<PyramidModelResult> process_pyramid_model(const std::shared_ptr<ov::Model>& original_model,
                                                        size_t model_idx,
                                                        size_t query_length,
                                                        size_t full_context_length,
                                                        const std::map<std::string, size_t>& past_key_sequence_dims,
                                                        const std::map<std::string, size_t>& past_value_sequence_dims);

// Helper function to validate model and extract necessary information for pyramid attention
std::optional<PyramidValidationResult> validate_and_setup_pyramid_attention(const std::shared_ptr<ov::Model>& model);

// Structure to hold SDPA pattern nodes
struct SDPAPatternNodes {
    std::shared_ptr<ov::Node> matmul1_node = nullptr;
    std::shared_ptr<ov::Node> matmul2_node = nullptr;
    std::shared_ptr<ov::Node> softmax_node = nullptr;
    std::shared_ptr<ov::Node> add_node = nullptr;

    bool isValid() const {
        return matmul1_node && matmul2_node && softmax_node && add_node;
    }
};

// Function to find SDPA pattern nodes in the model
SDPAPatternNodes findSDPAPatternNodes(const std::shared_ptr<ov::Model>& model);

// Function to remove empty KV inputs from model (optimization for model 0)
bool remove_empty_kv_inputs(std::shared_ptr<ov::Model> model);

// Function to find mask parameter by traversing from Add node
std::shared_ptr<ov::op::v0::Parameter> find_mask_parameter(const std::shared_ptr<ov::Node>& add_node);

// PyramidAttention structure definition
struct PyramidAttention {
    std::vector<struct Attention> _attentions;
    std::vector<std::shared_ptr<ov::Model>> _models;
    size_t _query_length = 0;
    size_t _full_context_length = 0;

    static std::optional<PyramidAttention> from(const std::shared_ptr<ov::Model>& model);
};

}  // namespace function
}  // namespace npuw
}  // namespace ov