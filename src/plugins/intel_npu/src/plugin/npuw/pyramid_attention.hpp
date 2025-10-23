// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#ifdef _WIN32
#    define NOMINMAX  // Prevent windows.h from defining min/max macros
// clang-format off
#    include <windows.h>
#    include <psapi.h>
// clang-format on
#    undef max  // Just in case
#    undef min  // Just in case
#else
#    include <unistd.h>

#    include <fstream>
#endif

#include "attention.hpp"

namespace ov {
namespace npuw {

// Helper function to get current process memory usage in KB
inline size_t get_process_memory_kb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / 1024;  // Convert bytes to KB
    }

    std::cout << "Failed to get process mem info" << std::endl;
    return 0;
#else
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.find("VmRSS:") == 0) {
            size_t pos = line.find_first_of("0123456789");
            if (pos != std::string::npos) {
                return std::stoul(line.substr(pos));  // Already in KB
            }
        }
    }

    std::cout << "Failed to get process mem info" << std::endl;
    return 0;
#endif
}

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

namespace compiled {

// Simplified pyramid attention parameter info
struct PyramidAttentionInfo {
    struct Param {
        std::size_t idx;  // function input index for this spatial parameter
        std::size_t dim;
    };
    std::vector<Param> params;
    std::size_t mask_idx = 0u;
    std::size_t query_size = 0u;  // Added for PositionIDs selector compatibility
};

// Compile-time pyramid attention information
struct PyramidAttention {
    std::vector<PyramidAttentionInfo> _attention_infos;
    std::vector<std::shared_ptr<ov::Model>> _models;
    std::vector<ov::SoPtr<ov::ICompiledModel>> _compiled_models;

    std::size_t query_size = 0u;
    std::size_t full_context_size = 0u;

    PyramidAttention() = delete;
    PyramidAttention(const function::PyramidAttention& d)
        : query_size(d._query_length),
          full_context_size(d._full_context_length) {
        NPUW_ASSERT(d._models.size() == d._attentions.size());

        // Memory measurement: record initial memory usage
        size_t initial_memory_kb = get_process_memory_kb();
        std::cout << "=== PyramidAttention Memory Tracking Start: " << initial_memory_kb << " KB RSS ===" << std::endl;

        for (size_t i = 0; i < d._attentions.size(); ++i) {
            size_t before_attention_kb = get_process_memory_kb();

            const auto& func_attn = d._attentions[i];
            const auto& model = d._models[i];

            PyramidAttentionInfo attention_info;
            // Extract parameters
            attention_info.params.reserve(func_attn._inputs.size());
            for (const auto& input : func_attn._inputs) {
                std::size_t p_idx = model->get_parameter_index(input.param);
                attention_info.params.push_back({p_idx, input.dim});
            }
            // Extract mask index and query size
            attention_info.mask_idx = model->get_parameter_index(func_attn._mask);
            attention_info.query_size = func_attn.query_len();

            size_t after_attention_kb = get_process_memory_kb();
            size_t attention_increase_kb =
                (after_attention_kb > before_attention_kb) ? (after_attention_kb - before_attention_kb) : 0;

            _attention_infos.push_back(attention_info);

            size_t before_model_kb = get_process_memory_kb();

            // This was suspected 2GB RSS increase
            _models.push_back(d._models[i]);

            size_t after_model_kb = get_process_memory_kb();
            size_t model_increase_kb = (after_model_kb > before_model_kb) ? (after_model_kb - before_model_kb) : 0;

            std::cout << "Model[" << i << "]: Attention info extraction increased RSS by " << attention_increase_kb
                      << " KB (" << (attention_increase_kb / 1024) << " MB)" << std::endl;
            std::cout << "Model[" << i << "]: push_back increased RSS by " << model_increase_kb << " KB ("
                      << (model_increase_kb / 1024) << " MB), total: " << after_model_kb << " KB" << std::endl;
        }

        size_t final_memory_kb = get_process_memory_kb();
        size_t total_increase_kb = (final_memory_kb > initial_memory_kb) ? (final_memory_kb - initial_memory_kb) : 0;
        std::cout << "=== PyramidAttention Memory Tracking End: Total increase = " << total_increase_kb << " KB ("
                  << (total_increase_kb / 1024) << " MB) ===" << std::endl;
    }

    // Return number of pyramid models
    size_t num_models() const {
        return _attention_infos.size();
    }
};

}  // namespace compiled

}  // namespace npuw
}  // namespace ov