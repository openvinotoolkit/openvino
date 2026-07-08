// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include "attention.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {

namespace function {

// Helper structs to hold validation and setup results.
// Two separate types make it impossible to accidentally access contiguous-mode
// fields in block mode or vice versa. The caller uses std::visit (or holds a
// std::variant) and is forced to handle each case explicitly.

struct PyramidValidationContiguousResult {
    size_t query_length = 0;
    size_t full_context_length = 0;
    size_t past_kv_length = 0;
    std::map<std::string, size_t> past_key_sequence_dims;
    std::map<std::string, size_t> past_value_sequence_dims;

    bool is_valid() const {
        return query_length > 0 && full_context_length > 0 && full_context_length >= query_length &&
               !past_key_sequence_dims.empty() && !past_value_sequence_dims.empty();
    }
};

struct PyramidValidationBlockResult {
    size_t query_length = 0;
    size_t full_context_length = 0;
    // Parameter indices for all N past-key/value blocks in the full (original) model,
    // in Concat input order (block_0 … block_{N-1}).
    std::vector<size_t> past_key_block_global_param_indices;
    std::vector<size_t> past_value_block_global_param_indices;

    bool is_valid() const {
        return query_length > 0 && full_context_length > 0 && full_context_length >= query_length &&
               past_key_block_global_param_indices.size() == past_value_block_global_param_indices.size();
    }
};

// validate_and_setup_pyramid_attention returns one of these two on success, nullopt on failure.
using PyramidValidationResult = std::variant<PyramidValidationContiguousResult, PyramidValidationBlockResult>;

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
    // Shared fields
    std::vector<Attention> _attentions;
    std::vector<std::shared_ptr<ov::Model>> _models;
    size_t _query_length = 0;
    size_t _full_context_length = 0;

    // Block mode only: global KV block parameter indices (block_0..block_N).
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

// Per-variant compiled metadata for contiguous KV cache mode.
// Populated by compiled::PyramidAttention constructor when no block split is present.
struct PyramidAttentionContiguousInfo {
    struct Param {
        std::size_t idx;  // function input index for this spatial parameter
        std::size_t dim;
    };

    std::size_t mask_idx = 0u;
    std::size_t query_size = 0u;
    std::size_t context_length = 0u;

    // Per-step KV slice descriptors used to bind past KV windows to pyramid variants.
    std::vector<Param> params;
};

// Per-variant compiled metadata for block-split KV cache mode
// (after SplitKVCacheIntoBlocks has been applied).
// Populated by compiled::PyramidAttention constructor when block indices are present.
struct PyramidAttentionBlockInfo {
    std::size_t mask_idx = 0u;
    std::size_t query_size = 0u;
    std::size_t context_length = 0u;

    // Precomputed direct lookup: global KV block param index → this variant's port index.
    // std::numeric_limits<size_t>::max() means this variant has no port for that block
    // (e.g. model[0] has 0 past blocks; model[i] has no port for blocks > i).
    // Enables O(1) binding in bind_function_input.
    std::unordered_map<size_t, size_t> past_key_block_port_map;
    std::unordered_map<size_t, size_t> past_value_block_port_map;

    // Precomputed set of this variant's KV block port indices.
    // Used by ensure_pyramid_requests to identify block ports during request setup.
    std::unordered_set<size_t> past_key_block_port_set;
    std::unordered_set<size_t> past_value_block_port_set;
};

// Per-variant compiled metadata: exactly one of contiguous or block mode is active.
// Note: PyramidAttentionContiguousInfo / PyramidAttentionBlockInfo are used directly
// by the PyramidAttentionContiguous and PyramidAttentionBlock subclasses below.

// Compile-time pyramid attention — abstract base + two concrete subclasses.
//
// PyramidAttentionContiguous holds one PyramidAttentionContiguousInfo per pyramid model;
// PyramidAttentionBlock       holds one PyramidAttentionBlockInfo    per pyramid model
//                             plus the global KV block parameter indices.
//
// The base class owns all shared runtime state (compiled models, context lengths,
// query/context sizes) and defines a virtual interface so consumers never need to
// branch on a flag or call std::get<>.
struct PyramidAttention {
    // Shared data
    std::vector<ov::SoPtr<ov::ICompiledModel>> _compiled_models;
    std::vector<std::size_t> _context_lengths;
    std::size_t query_size = 0u;
    std::size_t full_context_size = 0u;
    /// Whether non-last pyramid models were compiled with strided-input support.
    bool _can_use_tensor_view = false;
    /// Temporary storage for models pending compilation; cleared by set_compiled_models().
    std::vector<std::shared_ptr<ov::Model>> _models_to_compile;

    virtual ~PyramidAttention() = default;

    // Type discriminator
    virtual bool is_block_mode() const = 0;

    // Shared per-variant accessors
    virtual std::size_t mask_idx_at(size_t pyramid_id) const = 0;
    virtual std::size_t query_size_at(size_t pyramid_id) const = 0;

    // Block-mode accessors (contiguous subclass returns empty containers / zero)
    // Contiguous subclass returns empty containers / zero for all of these.
    virtual const std::unordered_set<size_t>& key_block_port_set_at(size_t pyramid_id) const = 0;
    virtual const std::unordered_set<size_t>& val_block_port_set_at(size_t pyramid_id) const = 0;
    virtual const std::unordered_map<size_t, size_t>& key_block_port_map_at(size_t pyramid_id) const = 0;
    virtual const std::unordered_map<size_t, size_t>& val_block_port_map_at(size_t pyramid_id) const = 0;
    virtual size_t num_key_blocks_global() const = 0;
    virtual size_t key_block_global_at(size_t block_idx) const = 0;
    virtual size_t val_block_global_at(size_t block_idx) const = 0;

    // Contiguous-mode KV param lookup
    // Returns the sequence dimension for input_idx in pyramid variant pyramid_id,
    // or nullopt when input_idx is not a KV param in that variant.
    // Block subclass always returns nullopt.
    virtual std::optional<std::size_t> kv_param_dim(size_t pyramid_id, size_t input_idx) const = 0;

    // Strides setup helper
    // Appends enable_strides_for input names for pyramid model 0 to 'out'.
    // No-op in block mode (block ports are bound directly, not via strided views).
    virtual void collect_strided_input_names(const ov::Model& model, std::string& out) const = 0;

    // Non-virtual shared methods
    void set_compiled_models(std::vector<ov::SoPtr<ov::ICompiledModel>>&& compiled_models);

    size_t num_models() const {
        return _context_lengths.size();
    }

    std::size_t get_context_length(size_t model_idx) const {
        return model_idx < _context_lengths.size() ? _context_lengths[model_idx] : 0;
    }

    // Static factory: constructs PyramidAttentionContiguous or PyramidAttentionBlock
    // depending on whether func_pyramid carries block KV indices.
    static std::shared_ptr<PyramidAttention> make(const function::PyramidAttention& func_pyramid);
};

// Concrete subclass for contiguous KV cache mode (legacy slice-and-copy path).
struct PyramidAttentionContiguous final : PyramidAttention {
    std::vector<PyramidAttentionContiguousInfo> _attention_infos;

    bool is_block_mode() const override {
        return false;
    }
    std::size_t mask_idx_at(size_t id) const override {
        return _attention_infos[id].mask_idx;
    }
    std::size_t query_size_at(size_t id) const override {
        return _attention_infos[id].query_size;
    }
    const std::unordered_set<size_t>& key_block_port_set_at(size_t) const override;
    const std::unordered_set<size_t>& val_block_port_set_at(size_t) const override;
    const std::unordered_map<size_t, size_t>& key_block_port_map_at(size_t) const override;
    const std::unordered_map<size_t, size_t>& val_block_port_map_at(size_t) const override;
    size_t num_key_blocks_global() const override {
        return 0u;
    }
    size_t key_block_global_at(size_t) const override {
        return 0u;
    }
    size_t val_block_global_at(size_t) const override {
        return 0u;
    }
    std::optional<std::size_t> kv_param_dim(size_t pyramid_id, size_t input_idx) const override;
    void collect_strided_input_names(const ov::Model& model, std::string& out) const override;
};

// Concrete subclass for block-split KV cache mode (after SplitKVCacheIntoBlocks).
struct PyramidAttentionBlock final : PyramidAttention {
    std::vector<PyramidAttentionBlockInfo> _attention_infos;
    /// Global (full-model) KV block parameter indices (block_0 … block_{N-1}).
    std::vector<size_t> past_key_block_global_param_indices;
    std::vector<size_t> past_value_block_global_param_indices;

    bool is_block_mode() const override {
        return true;
    }
    std::size_t mask_idx_at(size_t id) const override {
        return _attention_infos[id].mask_idx;
    }
    std::size_t query_size_at(size_t id) const override {
        return _attention_infos[id].query_size;
    }
    const std::unordered_set<size_t>& key_block_port_set_at(size_t id) const override {
        return _attention_infos[id].past_key_block_port_set;
    }
    const std::unordered_set<size_t>& val_block_port_set_at(size_t id) const override {
        return _attention_infos[id].past_value_block_port_set;
    }
    const std::unordered_map<size_t, size_t>& key_block_port_map_at(size_t id) const override {
        return _attention_infos[id].past_key_block_port_map;
    }
    const std::unordered_map<size_t, size_t>& val_block_port_map_at(size_t id) const override {
        return _attention_infos[id].past_value_block_port_map;
    }
    size_t num_key_blocks_global() const override {
        return past_key_block_global_param_indices.size();
    }
    size_t key_block_global_at(size_t i) const override {
        return past_key_block_global_param_indices[i];
    }
    size_t val_block_global_at(size_t i) const override {
        return past_value_block_global_param_indices[i];
    }
    std::optional<std::size_t> kv_param_dim(size_t, size_t) const override {
        return std::nullopt;
    }
    void collect_strided_input_names(const ov::Model&, std::string&) const override {}  // no-op
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
