// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "attention.hpp"
#include "sdpa_utils.hpp"

namespace ov {
namespace npuw {

namespace function {

// Helper struct to hold validation and setup results
struct PyramidValidationResult {
    // ── Shared ───────────────────────────────────────────────────────────────────
    size_t query_length = 0;
    size_t full_context_length = 0;
    // True when the model has already been processed by SplitKVCacheIntoBlocks.
    // Determines which of the two mode-specific field groups below is populated.
    bool is_block_split = false;

    // ── Contiguous mode only ─────────────────────────────────────────────────────
    // Empty when is_block_split == true.
    size_t past_kv_length = 0;
    std::map<std::string, size_t> past_key_sequence_dims;
    std::map<std::string, size_t> past_value_sequence_dims;

    static PyramidValidationResult make_contiguous(size_t query_len,
                                                   size_t ctx_len,
                                                   size_t past_kv_len,
                                                   std::map<std::string, size_t> key_seq_dims,
                                                   std::map<std::string, size_t> val_seq_dims) {
        PyramidValidationResult r;
        r.query_length = query_len;
        r.full_context_length = ctx_len;
        r.past_kv_length = past_kv_len;
        r.past_key_sequence_dims = std::move(key_seq_dims);
        r.past_value_sequence_dims = std::move(val_seq_dims);
        return r;
    }

    // ── Block mode only ──────────────────────────────────────────────────────────
    // Parameter indices for all N past-key/value blocks in the full (original) model.
    // Populated by validate_and_setup_pyramid_attention() so that from() can propagate
    // them into function::Attention (variant) and function::PyramidAttention (global).
    // Empty when is_block_split == false.
    std::vector<size_t> past_key_block_global_param_indices;
    std::vector<size_t> past_value_block_global_param_indices;

    static PyramidValidationResult make_block(size_t query_len,
                                              size_t ctx_len,
                                              std::vector<size_t> key_block_indices,
                                              std::vector<size_t> val_block_indices) {
        PyramidValidationResult r;
        r.query_length = query_len;
        r.full_context_length = ctx_len;
        r.is_block_split = true;
        r.past_key_block_global_param_indices = std::move(key_block_indices);
        r.past_value_block_global_param_indices = std::move(val_block_indices);
        return r;
    }

    // Validation helper
    bool is_valid() const {
        if (is_block_split) {
            // Block mode: dim maps are intentionally empty; only sizes must be sane.
            return query_length > 0 && full_context_length > 0 && full_context_length >= query_length;
        }
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

// Helper function to process a single pyramid model (clone, reshape, patch, optimize).
// When is_block_split is true the model has already been processed by SplitKVCacheIntoBlocks;
// in this case the function shrinks the KV Concat inputs rather than reshaping parameters.
std::optional<PyramidModelResult> process_pyramid_model(const std::shared_ptr<ov::Model>& original_model,
                                                        size_t model_idx,
                                                        size_t pyramid_step,
                                                        size_t query_length,
                                                        size_t full_past_kv_length,
                                                        size_t full_context_length,
                                                        const std::map<std::string, size_t>& past_key_sequence_dims,
                                                        const std::map<std::string, size_t>& past_value_sequence_dims,
                                                        bool is_block_split = false);

// Helper function to validate model and extract necessary information for pyramid attention
std::optional<PyramidValidationResult> validate_and_setup_pyramid_attention(const std::shared_ptr<ov::Model>& model);

// PyramidAttention structure definition
struct PyramidAttention {
    // ── Shared ───────────────────────────────────────────────────────────────────
    std::vector<Attention> _attentions;
    std::vector<std::shared_ptr<ov::Model>> _models;
    size_t _query_length = 0;
    size_t _full_context_length = 0;

    // ── Block mode only ──────────────────────────────────────────────────────────
    // Global (full-model) KV block parameter indices (block0..blockN).
    // Populated by from() when is_block_split; empty in contiguous mode.
    std::vector<size_t> past_key_block_global_param_indices;
    std::vector<size_t> past_value_block_global_param_indices;

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

    // ── Shared ───────────────────────────────────────────────────────────────────
    std::size_t mask_idx = 0u;
    std::size_t query_size = 0u;      // Added for PositionIDs selector compatibility
    std::size_t context_length = 0u;  // Context length this pyramid model supports

    // ── Contiguous mode only ─────────────────────────────────────────────────────
    // Used during both PREFILL and GENERATE to bind per-step KV slices. Empty in block mode.
    std::vector<Param> params;

    // ── Block mode only ──────────────────────────────────────────────────────────
    // Per-variant KV block parameter indices (block0..blockM, M <= N).
    // Copied from function::Attention::past_key/value_block_variant_param_indices by
    // the compiled::PyramidAttention constructor. Empty in contiguous mode.
    std::vector<size_t> past_key_block_variant_param_indices;
    std::vector<size_t> past_value_block_variant_param_indices;
};

// Compile-time pyramid attention information
struct PyramidAttention {
    // ── Shared ───────────────────────────────────────────────────────────────────
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

    // ── Block mode only ──────────────────────────────────────────────────────────
    // Global (full-model) KV block parameter indices (block0..blockN).
    // Copied from function::PyramidAttention::past_key/value_block_global_param_indices
    // by the constructor. Empty in contiguous mode.
    std::vector<size_t> past_key_block_global_param_indices;
    std::vector<size_t> past_value_block_global_param_indices;

    // Set compiled models after parallel compilation completes.
    // Clears _models_to_compile to free memory.
    // Block indices are already populated in the constructor from the graph.
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
