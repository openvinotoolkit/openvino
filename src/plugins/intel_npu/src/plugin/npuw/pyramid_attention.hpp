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
    bool data_left_aligned = false;
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
    // Total past KV length already cached across all N blocks (sum of each block's
    // sequence-axis size), analogous to PyramidValidationContiguousResult::past_kv_length.
    size_t past_kv_length = 0;
    // Parameter indices for all N past-key/value blocks in the full (original) model,
    // in Concat input order (block_0 … block_{N-1}).
    std::vector<size_t> past_key_block_global_param_indices;
    std::vector<size_t> past_value_block_global_param_indices;
    // Global (original/full model) parameter index of the attention mask. Same across all
    // pyramid variants (the mask parameter is never dropped, only reshaped).
    size_t global_mask_idx = 0;

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
    bool _data_left_aligned = false;

    // Block mode only: global KV block parameter indices (block_0..block_N).
    // Global (full-model) KV block parameter indices (block0..blockN).
    // Populated by from() when is_block_split; empty in contiguous mode.
    std::vector<size_t> past_key_block_global_param_indices;
    std::vector<size_t> past_value_block_global_param_indices;

    // Global (original/full model) parameter index of the attention mask. Same across all
    // pyramid variants (the mask parameter is never dropped, only reshaped). Populated by
    // from() in both contiguous and block modes.
    size_t global_mask_idx = 0;

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

    std::size_t mask_idx_local = 0u;
    std::size_t query_size = 0u;
    std::size_t context_length = 0u;

    // Per-step KV slice descriptors used to bind past KV windows to pyramid variants.
    std::vector<Param> params;
};

// Per-variant compiled metadata for block-split KV cache mode
// (after SplitKVCacheIntoBlocks has been applied).
// Populated by compiled::PyramidAttention constructor when block indices are present.
//
// Index-space convention (applies to every field below): all *keys* / bare indices are in
// the GLOBAL (canonical/main model) index space — the same space as bind_function_input's
// input_idx — because that's the only index space callers ever have on hand. Only the
// *values* used to actually call set_tensor() on this variant's compiled model (map values,
// mask_idx_local) are in this variant's own LOCAL (shifted-by-removed-blocks) index space.
struct PyramidAttentionBlockInfo {
    // LOCAL (this variant's) mask port index — an index into this variant's own compiled
    // model, NOT the global index. Compare against PyramidAttention::global_mask_idx, never
    // against a caller-supplied global input_idx.
    std::size_t mask_idx_local = 0u;
    std::size_t query_size = 0u;
    std::size_t context_length = 0u;

    // Precomputed set of this variant's LOCAL KV block port indices.
    // Used by ensure_pyramid_requests to identify block ports during request setup.
    std::unordered_set<size_t> past_key_block_port_set;
    std::unordered_set<size_t> past_value_block_port_set;

    // GLOBAL input index → this variant's LOCAL input index, for ALL retained parameters
    // (mask, KV blocks, and everything else). Built from
    // function::Attention::global_to_local_param_idx. Required because surplus KV block
    // Parameters are removed per variant, which shifts the index of every remaining
    // Parameter — unlike contiguous mode, where all variants keep the same parameter
    // order/count as the main model and indices can be reused as-is.
    //
    // A GLOBAL KV block index that this variant dropped has NO entry here (rather than an
    // explicit sentinel): callers must first check PyramidAttention::is_key_block_global_idx()
    // / is_value_block_global_idx() to tell "dropped block" apart from "not a block at all"
    // when a lookup misses. Enables O(1) binding in bind_function_input.
    std::unordered_map<size_t, size_t> param_port_map;
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

    bool _data_left_aligned = false;
    /// Temporary storage for models pending compilation; cleared by set_compiled_models().
    std::vector<std::shared_ptr<ov::Model>> _models_to_compile;

    // GLOBAL (original/full model) parameter index of the attention mask. Identical across
    // all pyramid variants (contiguous: LOCAL index equals GLOBAL index since no parameters
    // are ever dropped; block: propagated from function::PyramidAttention::global_mask_idx).
    // Use this — never mask_idx_local_at() — when comparing against a caller-supplied global
    // input_idx.
    std::size_t global_mask_idx = 0u;

    virtual ~PyramidAttention() = default;

    // Type discriminator
    virtual bool is_block_mode() const = 0;

    // Shared per-variant accessors. Returns the LOCAL (this variant's) mask index — do not
    // compare it against a global input_idx; use global_mask_idx for that.
    virtual std::size_t mask_idx_local_at(size_t pyramid_id) const = 0;
    virtual std::size_t query_size_at(size_t pyramid_id) const = 0;

    // Block-mode accessors (contiguous subclass returns empty containers / zero)
    // Contiguous subclass returns empty containers / zero for all of these.
    virtual const std::unordered_set<size_t>& key_block_port_set_at(size_t pyramid_id) const = 0;
    virtual const std::unordered_set<size_t>& val_block_port_set_at(size_t pyramid_id) const = 0;
    virtual size_t num_key_blocks_global() const = 0;
    virtual size_t key_block_global_at(size_t block_idx) const = 0;
    virtual size_t val_block_global_at(size_t block_idx) const = 0;

    // Block-mode only: true if 'idx' (a GLOBAL parameter index) is one of the KV block
    // Parameters in the full/original model — regardless of whether this specific variant
    // still has a port for it. Used to distinguish "this variant dropped a known KV block"
    // (param_port_map_at() has no entry, but this returns true) from "input isn't a KV block
    // at all" (both would return false/miss). Contiguous subclass always returns false.
    virtual bool is_key_block_global_idx(size_t idx) const = 0;
    virtual bool is_value_block_global_idx(size_t idx) const = 0;

    // Contiguous-mode KV param lookup
    // Returns the sequence dimension for input_idx in pyramid variant pyramid_id,
    // or nullopt when input_idx is not a KV param in that variant.
    // Block subclass always returns nullopt.
    virtual std::optional<std::size_t> kv_param_dim(size_t pyramid_id, size_t input_idx) const = 0;

    // Block-mode only: GLOBAL input index → this variant's LOCAL parameter port index, for
    // ALL parameters (see PyramidAttentionBlockInfo::param_port_map). Contiguous subclass
    // returns an empty map — its variants share the exact same parameter set/order as the
    // main model, so input_idx is already a valid LOCAL index and this lookup is unnecessary.
    virtual const std::unordered_map<size_t, size_t>& param_port_map_at(size_t pyramid_id) const = 0;

    // Strides setup helper
    // Appends enable_strides_for input names for pyramid model 0 to 'out'.
    // No-op in block mode (block ports are bound directly, not via strided views).
    virtual void collect_strided_input_names(const ov::Model& model, std::string& out) const = 0;

    // Validates that all port indices in _attention_infos are within bounds of the
    // corresponding compiled model's inputs(). Throws ov::Exception on violation.
    // Must be called after _compiled_models is fully populated (import path).
    virtual void validate_port_indices() const = 0;

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
    std::size_t mask_idx_local_at(size_t id) const override {
        return _attention_infos[id].mask_idx_local;
    }
    std::size_t query_size_at(size_t id) const override {
        return _attention_infos[id].query_size;
    }
    const std::unordered_set<size_t>& key_block_port_set_at(size_t) const override;
    const std::unordered_set<size_t>& val_block_port_set_at(size_t) const override;
    size_t num_key_blocks_global() const override {
        return 0u;
    }
    size_t key_block_global_at(size_t) const override {
        return 0u;
    }
    size_t val_block_global_at(size_t) const override {
        return 0u;
    }
    bool is_key_block_global_idx(size_t) const override {
        return false;
    }
    bool is_value_block_global_idx(size_t) const override {
        return false;
    }
    std::optional<std::size_t> kv_param_dim(size_t pyramid_id, size_t input_idx) const override;
    const std::unordered_map<size_t, size_t>& param_port_map_at(size_t) const override;
    void collect_strided_input_names(const ov::Model& model, std::string& out) const override;
    void validate_port_indices() const override;
};

// Concrete subclass for block-split KV cache mode (after SplitKVCacheIntoBlocks).
struct PyramidAttentionBlock final : PyramidAttention {
    std::vector<PyramidAttentionBlockInfo> _attention_infos;
    /// Global (full-model) KV block parameter indices (block_0 … block_{N-1}).
    std::vector<size_t> past_key_block_global_param_indices;
    std::vector<size_t> past_value_block_global_param_indices;
    /// O(1) membership views over the vectors above, for is_key/value_block_global_idx().
    std::unordered_set<size_t> _key_block_global_set;
    std::unordered_set<size_t> _value_block_global_set;

    bool is_block_mode() const override {
        return true;
    }
    std::size_t mask_idx_local_at(size_t id) const override {
        return _attention_infos[id].mask_idx_local;
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
    size_t num_key_blocks_global() const override {
        return past_key_block_global_param_indices.size();
    }
    size_t key_block_global_at(size_t i) const override {
        return past_key_block_global_param_indices[i];
    }
    size_t val_block_global_at(size_t i) const override {
        return past_value_block_global_param_indices[i];
    }
    bool is_key_block_global_idx(size_t idx) const override {
        return _key_block_global_set.count(idx) != 0;
    }
    bool is_value_block_global_idx(size_t idx) const override {
        return _value_block_global_set.count(idx) != 0;
    }
    std::optional<std::size_t> kv_param_dim(size_t, size_t) const override {
        return std::nullopt;
    }
    const std::unordered_map<size_t, size_t>& param_port_map_at(size_t id) const override {
        return _attention_infos[id].param_port_map;
    }
    void collect_strided_input_names(const ov::Model&, std::string&) const override {}  // no-op
    void validate_port_indices() const override;
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

// Define pyramid model selection based on position ids.
// Handles the regular (contiguous) case where the KV update step matches the query
// length, e.g. the SDPADecomposed pattern.
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

// Define pyramid model selection based on position ids for attention patterns where the
// KV update step (pyramid_step) differs from the query length - e.g. QuantizedSDPAWithGlobalMask,
// where a large query is matched against a smaller-granularity KV cache update.
// Equivalent to PositionIDs, except past/current sequence length are derived from
// pyramid_step instead of query_size. Constructed only via PositionIDs::find().
class GlobalPositionIDs final : public Selector {
    std::size_t m_position_ids_idx = 0u;
    int64_t m_current_length = 0;
    int64_t m_past_length = 0;
    std::size_t m_query_size = 0u;
    std::size_t m_pyramid_step = 0u;

    // Store pyramid attention reference for pyramid model selection
    const compiled::PyramidAttention* m_pyramid_attention = nullptr;

    const ov::ISyncInferRequest& m_rq;

    GlobalPositionIDs(std::size_t param_idx, const compiled::PyramidAttention& d, const ov::ISyncInferRequest& rq);
    void prepare(int64_t past_len) override;
    int64_t length() const override;
    int64_t past_length() const override;

    friend class PositionIDs;  // constructed only via PositionIDs::find()
};

}  // namespace pyramid_attention
}  // namespace runtime

}  // namespace npuw
}  // namespace ov
