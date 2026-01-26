// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "attention.hpp"

namespace ov {
namespace npuw {

namespace function {

// Helper struct to hold validation and setup results
struct PyramidValidationResult {
    size_t query_length = 0;
    size_t past_kv_length = 0;
    size_t full_context_length = 0;
    std::map<std::string, size_t> past_key_sequence_dims;
    std::map<std::string, size_t> past_value_sequence_dims;

    // Validation helper
    bool is_valid() const {
        return query_length > 0 && full_context_length > 0 && full_context_length >= query_length &&
               !past_key_sequence_dims.empty() && !past_value_sequence_dims.empty();
    }
};

// Helper struct to hold model processing result
struct PyramidModelResult {
    std::shared_ptr<ov::Model> model;
    ov::npuw::function::Attention attention;

    bool is_valid() const {
        return model != nullptr;
    }
};

// Helper function to create Attention instance from a model
std::optional<ov::npuw::function::Attention> create_attention_from_model(
    const std::shared_ptr<ov::Model>& model,
    const std::map<std::string, size_t>& past_key_sequence_dims,
    const std::map<std::string, size_t>& past_value_sequence_dims);

// Helper function to process a single pyramid model (clone, reshape, patch, optimize)
std::optional<PyramidModelResult> process_pyramid_model(const std::shared_ptr<ov::Model>& original_model,
                                                        size_t model_idx,
                                                        size_t pyramid_step,
                                                        size_t query_length,
                                                        size_t full_past_kv_length,
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
    std::shared_ptr<ov::Node> past_key_param_node = nullptr;
    std::shared_ptr<ov::Node> past_value_param_node = nullptr;
    std::shared_ptr<ov::Node> past_key_concat_node = nullptr;
    std::shared_ptr<ov::Node> past_value_concat_node = nullptr;

    bool is_valid() const {
        return matmul1_node && matmul2_node && softmax_node && add_node && past_key_param_node &&
               past_value_param_node && past_key_concat_node && past_value_concat_node;
    }

    // Log pattern information for debugging
    void log_pattern() const {
        LOG_DEBUG("SDPA Pattern nodes:");
        LOG_DEBUG("  MatMul1: " << (matmul1_node ? matmul1_node->get_friendly_name() : "null"));
        LOG_DEBUG("  Add: " << (add_node ? add_node->get_friendly_name() : "null"));
        LOG_DEBUG("  Softmax: " << (softmax_node ? softmax_node->get_friendly_name() : "null"));
        LOG_DEBUG("  MatMul2: " << (matmul2_node ? matmul2_node->get_friendly_name() : "null"));
        LOG_DEBUG("  Key Concat: " << (past_key_concat_node ? past_key_concat_node->get_friendly_name() : "null"));
        LOG_DEBUG(
            "  Value Concat: " << (past_value_concat_node ? past_value_concat_node->get_friendly_name() : "null"));
    }
};

// Function to find SDPA pattern nodes in the model
SDPAPatternNodes find_sdpa_pattern_nodes(const std::shared_ptr<ov::Model>& model);

// Function to find mask parameter by traversing from Add node
std::shared_ptr<ov::op::v0::Parameter> find_mask_parameter(const std::shared_ptr<ov::Node>& add_node);

// PyramidAttention structure definition
struct PyramidAttention {
    std::vector<Attention> _attentions;
    std::vector<std::shared_ptr<ov::Model>> _models;
    size_t _query_length = 0;
    size_t _full_context_length = 0;

    // Validation helpers
    bool is_valid() const {
        return !_models.empty() && _models.size() == _attentions.size() && _query_length > 0 &&
               _full_context_length > 0;
    }

    size_t num_models() const {
        return _models.size();
    }

    // Factory method
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
    std::size_t query_size = 0u;      // Added for PositionIDs selector compatibility
    std::size_t context_length = 0u;  // Context length this pyramid model supports
};

// Compile-time pyramid attention information
struct PyramidAttention {
    std::vector<PyramidAttentionInfo> _attention_infos;
    std::vector<ov::SoPtr<ov::ICompiledModel>> _compiled_models;
    std::vector<std::size_t> _context_lengths;

    std::size_t query_size = 0u;
    std::size_t full_context_size = 0u;

    // Store models temporarily for compilation - cleared after compilation completes in set_compiled_models()
    std::vector<std::shared_ptr<ov::Model>> _models_to_compile;

    PyramidAttention() = default;

    // Constructor that extracts metadata and stores models for compilation
    // Compiled models are set later via set_compiled_models()
    explicit PyramidAttention(const function::PyramidAttention& func_pyramid);

    // Set compiled models after parallel compilation completes
    // Also clears _models_to_compile to free memory
    void set_compiled_models(std::vector<ov::SoPtr<ov::ICompiledModel>>&& compiled_models);

    // Return number of pyramid models
    size_t num_models() const {
        return _attention_infos.size();
    }

    // Get context length for a specific model
    std::size_t get_context_length(size_t model_idx) const {
        return model_idx < _context_lengths.size() ? _context_lengths[model_idx] : 0;
    }
};

}  // namespace compiled

namespace runtime {
namespace pyramid_attention {

// A base class to decide pyramid model selection
class Selector {
public:
    enum class Case { PREFILL, GENERATE, UNKNOWN };

    using Ptr = std::shared_ptr<Selector>;
    virtual ~Selector() = default;
    virtual void prepare(int64_t past_len) = 0;
    virtual int64_t length() const = 0;
    virtual int64_t past_length() const = 0;

    // Getter for the selected pyramid model ID (updated by prepare())
    std::size_t pyramid_id() const {
        return m_pyramid_id;
    }

    Case this_case() const {
        return m_case;
    }

protected:
    Case m_case = Case::UNKNOWN;
    std::size_t m_pyramid_id = 0;  // Selected pyramid model ID, updated by prepare()
};

// No dynamic dispatch - just use the largest pyramid model
class All final : public Selector {
    std::size_t m_pyramid_count = 0;

public:
    explicit All(std::size_t pyramid_count) : m_pyramid_count(pyramid_count) {}

    void prepare(int64_t past_len) override {
        // Always use the largest pyramid model (last one)
        m_pyramid_id = m_pyramid_count > 0 ? m_pyramid_count - 1 : 0;
    }
    int64_t length() const override {
        return -1;
    }
    int64_t past_length() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
};

// Define pyramid model selection based on position ids
class PositionIDs final : public Selector {
    std::size_t m_position_ids_idx = 0u;
    int64_t m_current_length = 0;
    int64_t m_past_length = 0;
    std::size_t m_query_size = 0u;

    // Store pyramid attention reference for pyramid model selection
    const compiled::PyramidAttention* m_pyramid_attention = nullptr;

    const ov::ISyncInferRequest& m_rq;

    PositionIDs(std::size_t param_idx, const compiled::PyramidAttention& d, const ov::ISyncInferRequest& rq);
    void prepare(int64_t past_len) override;
    int64_t length() const override;
    int64_t past_length() const override;

public:
    static Selector::Ptr find(const compiled::PyramidAttention& d, const ov::ISyncInferRequest& rq);
};

}  // namespace pyramid_attention
}  // namespace runtime

}  // namespace npuw
}  // namespace ov
