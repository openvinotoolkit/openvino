// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "attention.hpp"
#include "openvino/core/except.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {

// SDPA (Scaled Dot-Product Attention) input tensor identifiers
// Represents the standardized input layout for SDPA operations
// Defined at namespace level for use in both function and compiled namespaces
enum class SDPAInputId : uint8_t {
    PAST_KEY = 0,        // Historical key cache tensor
    PAST_VALUE = 1,      // Historical value cache tensor
    QUERY = 2,           // Query tensor for current iteration
    PRESENT_KEY = 3,     // Current key tensor (new tokens)
    ATTENTION_MASK = 4,  // Attention mask tensor
    PRESENT_VALUE = 5,   // Current value tensor (new tokens)

    // Sentinel value for enum range
    COUNT
};

// HFA Tile Model input tensor identifiers
// Represents the input layout for Host Flash Attention tile models
// Input names: [past_acc, past_max, past_d, k_tile, v_tile, q, mask_tile]
enum class HFATileInputId : uint8_t {
    PAST_ACC = 0,   // Accumulated attention output from previous tiles
    PAST_MAX = 1,   // Maximum values from previous tiles (for numerical stability)
    PAST_D = 2,     // Normalization denominator from previous tiles
    K_TILE = 3,     // Current K (key) tile slice
    V_TILE = 4,     // Current V (value) tile slice
    Q = 5,          // Query tensor (full, not tiled)
    MASK_TILE = 6,  // Current attention mask tile slice

    // Sentinel value for enum range
    COUNT
};

namespace function {

// HostFlashAttention structure definition
struct HostFlashAttention {
    // Original SDPA model (for parameter extraction)
    std::shared_ptr<ov::Model> _original_model;

    // Tiled model for flash attention execution (regular tiles)
    std::shared_ptr<ov::Model> _tile_model;

    // Final tiled model for flash attention execution (with division and transpose)
    std::shared_ptr<ov::Model> _final_tile_model;

    // Attention metadata from original SDPA model (mask, past key/value inputs)
    Attention _sdpa_attention;

    // Tile configuration
    int64_t _tile_size = 1024;  // Default K/V tile size

    // Input/Output mapping for tiled execution
    // Inputs: [past_acc, past_max, past_d, k_tile, v_tile, q, mask_tile]
    // Outputs: [new_acc, new_max, new_d]

    // Total KV cache size for tiling
    int64_t _kv_cache_size = 0;

    // SDPA model parameter index mapping
    // Maps semantic SDPA parameter IDs to actual parameter indices in the original SDPA model
    // This is created during pattern analysis in from() method
    std::map<SDPAInputId, std::size_t> _sdpa_param_index_map;

    // HFA Tile Model parameter index mapping
    // Maps semantic tile parameter IDs to actual parameter indices in the tile model
    // This is created after tile model generation in from() method
    std::map<HFATileInputId, std::size_t> _tile_param_index_map;

    // Validation helpers
    bool is_valid() const {
        return _tile_model != nullptr && _final_tile_model != nullptr && _tile_size > 0 && _kv_cache_size > 0;
    }

    // Factory method
    static std::optional<HostFlashAttention> from(const std::shared_ptr<ov::Model>& model);
};

}  // namespace function

namespace compiled {

// Simplified host flash attention parameter info
// Contains parameter indices from the original SDPA model
struct HostFlashAttentionInfo {
    struct Param {
        std::size_t idx;  // parameter index in original SDPA model
        std::size_t dim;  // dimension index for sequence length
    };
    std::vector<Param> params;    // past key/value parameters from original SDPA
    std::size_t mask_idx = 0u;    // mask parameter index in original SDPA model
    std::size_t query_size = 0u;  // query size for selector compatibility

    // Mapping from SDPA parameter identifier to actual parameter index in original SDPA model
    // This allows accessing SDPA model parameters by semantic name rather than hardcoded indices
    // Populated from function::HostFlashAttention::_sdpa_param_index_map
    std::map<SDPAInputId, std::size_t> sdpa_param_index_map;

    // Mapping from HFA Tile parameter identifier to actual parameter index in tile model
    // This allows accessing tile model parameters by semantic name
    // Populated from function::HostFlashAttention::_tile_param_index_map
    std::map<HFATileInputId, std::size_t> tile_param_index_map;
};

// Compile-time host flash attention information
struct HostFlashAttention {
    // Models to compile (will be cleared after compilation)
    std::shared_ptr<ov::Model> _tile_model_to_compile;
    std::shared_ptr<ov::Model> _final_tile_model_to_compile;

    // Compiled tile model for NPU execution (regular tiles)
    ov::SoPtr<ov::ICompiledModel> _compiled_tile_model;

    // Compiled FINAL tile model for NPU execution (with division and transpose)
    ov::SoPtr<ov::ICompiledModel> _compiled_final_tile_model;

    // Attention parameter info from original SDPA model (not from tile models)
    HostFlashAttentionInfo _sdpa_attention_info;

    // Tile configuration
    int64_t _tile_size = 1024;
    int64_t _kv_cache_size = 0;

    HostFlashAttention() = default;

    // Constructor that extracts metadata
    explicit HostFlashAttention(const function::HostFlashAttention& func_hfa);

    // Set the compiled tile model and clear the model to compile
    void set_compiled_tile_model(ov::SoPtr<ov::ICompiledModel> compiled_model) {
        _compiled_tile_model = std::move(compiled_model);
        _tile_model_to_compile.reset();  // Free memory after compilation
    }

    // Set the compiled FINAL tile model and clear the model to compile
    void set_compiled_final_tile_model(ov::SoPtr<ov::ICompiledModel> compiled_model) {
        _compiled_final_tile_model = std::move(compiled_model);
        _final_tile_model_to_compile.reset();  // Free memory after compilation
    }

    bool is_valid() const {
        return _compiled_tile_model != nullptr && _compiled_final_tile_model != nullptr && _tile_size > 0 &&
               _kv_cache_size > 0;
    }
};

}  // namespace compiled

namespace runtime {
namespace host_flash_attention {

// A base class to decide host flash attention execution
class Selector {
public:
    enum class Case { PREFILL, GENERATE, UNKNOWN };

    using Ptr = std::shared_ptr<Selector>;
    virtual ~Selector() = default;
    virtual void prepare(int64_t past_len) = 0;
    virtual int64_t length() const = 0;
    virtual int64_t past_length() const = 0;

    Case this_case() const {
        return m_case;
    }

protected:
    Case m_case = Case::UNKNOWN;
};

// No dynamic dispatch - just use default execution
class All final : public Selector {
public:
    void prepare(int64_t past_len) override {}
    int64_t length() const override {
        return -1;
    }
    int64_t past_length() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
};

// Define execution selection based on position ids
class PositionIDs final : public Selector {
    std::size_t m_position_ids_idx = 0u;
    int64_t m_current_length = 0;
    int64_t m_past_length = 0;
    std::size_t m_query_size = 0u;

    const ov::ISyncInferRequest& m_rq;

    PositionIDs(std::size_t param_idx, std::size_t query_size, const ov::ISyncInferRequest& rq);
    void prepare(int64_t past_len) override;
    int64_t length() const override;
    int64_t past_length() const override;

public:
    static Selector::Ptr find(std::size_t query_size, const ov::ISyncInferRequest& rq);
};

}  // namespace host_flash_attention
}  // namespace runtime

}  // namespace npuw
}  // namespace ov
