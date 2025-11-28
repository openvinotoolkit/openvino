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

constexpr int64_t DEFAULT_TILE_SIZE = 1024;

// Helper functions to convert enum values to string representations for logging/debugging
inline const char* sdpa_input_id_to_string(SDPAInputId id) {
    switch (id) {
    case SDPAInputId::PAST_KEY:
        return "PAST_KEY";
    case SDPAInputId::PAST_VALUE:
        return "PAST_VALUE";
    case SDPAInputId::QUERY:
        return "QUERY";
    case SDPAInputId::PRESENT_KEY:
        return "PRESENT_KEY";
    case SDPAInputId::ATTENTION_MASK:
        return "ATTENTION_MASK";
    case SDPAInputId::PRESENT_VALUE:
        return "PRESENT_VALUE";
    default:
        return "UNKNOWN";
    }
}

inline const char* hfa_tile_input_id_to_string(HFATileInputId id) {
    switch (id) {
    case HFATileInputId::PAST_ACC:
        return "PAST_ACC";
    case HFATileInputId::PAST_MAX:
        return "PAST_MAX";
    case HFATileInputId::PAST_D:
        return "PAST_D";
    case HFATileInputId::K_TILE:
        return "K_TILE";
    case HFATileInputId::V_TILE:
        return "V_TILE";
    case HFATileInputId::Q:
        return "Q";
    case HFATileInputId::MASK_TILE:
        return "MASK_TILE";
    default:
        return "UNKNOWN";
    }
}

namespace function {

// HostFlashAttention structure definition
struct HostFlashAttention {
    // Tiled model for flash attention execution (regular tiles)
    std::shared_ptr<ov::Model> _tile_model;

    // Final tiled model for flash attention execution (with division and transpose)
    std::shared_ptr<ov::Model> _final_tile_model;

    // Tile configuration
    int64_t _tile_size = 1024;  // K/V tile size for flash attention chunking

    // Query sequence length (extracted from Q shape[2])
    // Used for selector compatibility and runtime decision-making
    std::size_t _query_size = 0;

    // Sequence dimension indices for K and V tensors
    // These indicate which dimension is the sequence/cache dimension in past_key and past_value tensors
    // Extracted from Concat operations in the SDPA pattern
    std::size_t _k_seq_dim = 0;
    std::size_t _v_seq_dim = 0;

    // SDPA model parameter index mapping
    // Maps semantic SDPA parameter IDs (QUERY, PAST_KEY, etc.) to actual parameter indices
    // This is created during pattern analysis in from() method
    std::map<SDPAInputId, std::size_t> _sdpa_param_index_map;

    // Tile model parameter index mapping
    // Maps tile parameter IDs (PAST_ACC, K_TILE, Q, etc.) to actual input indices
    // Tile model I/O: Inputs[past_acc, past_max, past_d, k_tile, v_tile, q, mask_tile]
    //                 Outputs[acc, max, d] for regular tiles or [output] for final tile
    // This is created after tile model generation in from() method
    std::map<HFATileInputId, std::size_t> _tile_param_index_map;

    // Validation helpers
    bool is_valid() const {
        return _tile_model != nullptr && _final_tile_model != nullptr && _tile_size > 0;
    }

    // Factory method
    static std::optional<HostFlashAttention> from(const std::shared_ptr<ov::Model>& model);
};

}  // namespace function

namespace compiled {

// Simplified host flash attention parameter info
// Contains parameter indices from the original SDPA model
struct HostFlashAttentionInfo {
    std::size_t _query_size = 0u;  // query size for selector compatibility

    // Sequence dimension indices for K and V tensors in the original SDPA model
    // These indicate which dimension is the sequence/cache dimension in past_key and past_value tensors
    // Copied from function::HostFlashAttention::_k_seq_dim and _v_seq_dim
    std::size_t _k_seq_dim = 0u;
    std::size_t _v_seq_dim = 0u;

    // Mapping from SDPA parameter identifier to actual parameter index in original SDPA model
    // This allows accessing SDPA model parameters by semantic name rather than hardcoded indices
    // Populated from function::HostFlashAttention::_sdpa_param_index_map
    std::map<SDPAInputId, std::size_t> _sdpa_param_index_map;

    // Mapping from HFA Tile parameter identifier to actual parameter index in tile model
    // This allows accessing tile model parameters by semantic name
    // Populated from function::HostFlashAttention::_tile_param_index_map
    std::map<HFATileInputId, std::size_t> _tile_param_index_map;
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
        return _compiled_tile_model != nullptr && _compiled_final_tile_model != nullptr && _tile_size > 0;
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
    virtual int64_t context_length() const = 0;

    Case this_case() const {
        return _case;
    }

protected:
    Case _case = Case::UNKNOWN;
};

// Selector that processes all tiles unconditionally (no dynamic range optimization)
class All final : public Selector {
public:
    void prepare(int64_t past_len) override {}

    int64_t context_length() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
};

// Define execution selection based on position ids
class PositionIDs final : public Selector {
    std::size_t _position_ids_idx = 0u;
    int64_t _current_length = 0;
    int64_t _past_length = 0;
    std::size_t _query_size = 0u;

    const ov::ISyncInferRequest& _rq;

    PositionIDs(std::size_t param_idx, std::size_t query_size, const ov::ISyncInferRequest& rq);
    void prepare(int64_t past_len) override;
    int64_t context_length() const override;

public:
    static Selector::Ptr find(std::size_t query_size, const ov::ISyncInferRequest& rq);
};

}  // namespace host_flash_attention
}  // namespace runtime

}  // namespace npuw
}  // namespace ov
