// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <optional>

#include "openvino/core/except.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/itensor.hpp"
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

// HFA Regular Tile Model output tensor identifiers
// Represents the output layout for regular (non-final) tile models
// Output names: [acc, maxx, d]
enum class HFATileOutputId : uint8_t {
    ACC = 0,   // Accumulated attention output
    MAXX = 1,  // Maximum values for numerical stability
    D = 2,     // Normalization denominator

    // Sentinel value for enum range
    COUNT
};

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

inline const char* hfa_tile_output_id_to_string(HFATileOutputId id) {
    switch (id) {
    case HFATileOutputId::ACC:
        return "ACC";
    case HFATileOutputId::MAXX:
        return "MAXX";
    case HFATileOutputId::D:
        return "D";
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
    int64_t _tile_size = 0;  // K/V tile size for flash attention chunking

    // Query sequence length (extracted from Q shape[2])
    // Used for selector compatibility and runtime decision-making
    std::size_t _query_size = 0;

    // Context size (extracted from K concat output shape - kv_cache_size dimension)
    // Represents the total KV cache length available for attention computation
    std::size_t _context_size = 0;

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

    // Tile model output index mapping
    // Maps tile output IDs (UPDATED_ACC, UPDATED_MAX, UPDATED_D) to actual output indices
    // Only applicable to regular tile model (final tile has single output at index 0)
    // This is created after tile model generation in from() method
    std::map<HFATileOutputId, std::size_t> _tile_output_index_map;

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
    std::size_t _context_size = 0u;

    // Sequence dimension indices for K and V tensors in the original SDPA model
    // These indicate which dimension is the sequence/cache dimension in past_key and past_value tensors
    // Copied from function::HostFlashAttention::_k_seq_dim and _v_seq_dim
    std::size_t _k_seq_dim = 0u;
    std::size_t _v_seq_dim = 0u;

    // NOTE: SDPA parameter map is not stored to save memory.
    // Indices are pre-cached below during compilation.

    // Pre-cached SDPA parameter indices
    struct {
        std::size_t query = 0u;
        std::size_t past_key = 0u;
        std::size_t past_value = 0u;
        std::size_t present_key = 0u;
        std::size_t present_value = 0u;
        std::size_t attention_mask = 0u;
    } _sdpa_indices;

    // Pre-cached tile input indices
    struct {
        std::size_t q = 0u;
        std::size_t k = 0u;
        std::size_t v = 0u;
        std::size_t mask = 0u;
        std::size_t acc = 0u;
        std::size_t max = 0u;
        std::size_t d = 0u;
    } _tile_input_indices;

    // Pre-cached tile output indices
    struct {
        std::size_t acc = 0u;
        std::size_t max = 0u;
        std::size_t d = 0u;
    } _tile_output_indices;
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
    int64_t _tile_size = 0;

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

// Key for HFA tiled attention mask cache
struct HFATileMaskKey {
    ov::SoPtr<ov::ITensor> mask_tensor;  // Original mask tensor
    int64_t mask_offset;                 // Offset in the mask sequence dimension
    int64_t tile_length;                 // Length of the tile

    bool operator<(const HFATileMaskKey& other) const {
        if (mask_tensor._ptr != other.mask_tensor._ptr)
            return mask_tensor._ptr < other.mask_tensor._ptr;
        if (mask_offset != other.mask_offset)
            return mask_offset < other.mask_offset;
        return tile_length < other.tile_length;
    }
};

/// HFA runtime context - manages cache and buffers for tiled attention.
/// @warning NOT thread-safe. Each inference request must have its own instance.
struct HFARuntimeContext {
    // ============================================================================
    // Type Aliases
    // ============================================================================

    /// Memory allocator: (type, shape, device) -> tensor
    using AllocatorFn =
        std::function<ov::SoPtr<ov::ITensor>(const ov::element::Type&, const ov::Shape&, const std::string&)>;

    /// State tensors for accumulation: acc, max, sum
    struct StateBuffers {
        ov::SoPtr<ov::ITensor> acc;  // Accumulated output
        ov::SoPtr<ov::ITensor> max;  // Max values for stability
        ov::SoPtr<ov::ITensor> sum;  // Normalization denominator
    };

    // ============================================================================
    // Mask Cache Optimization
    // ============================================================================

    /// Cached mask tiles: (tensor, offset, length) -> tile
    std::map<HFATileMaskKey, ov::SoPtr<ov::ITensor>> m_mask_tile_cache;

    /// Pre-allocated buffers for mask extraction on cache miss
    std::vector<ov::SoPtr<ov::ITensor>> m_mask_tile_buffers;

    // ============================================================================
    // State Double-Buffering Optimization
    // ============================================================================

    /// Two state buffers: [current, next] (swap after each inference)
    std::optional<std::array<StateBuffers, 2>> m_state_buffers;

    /// Current active buffer index (0 or 1)
    size_t m_current_buffer_idx = 0;

    // ============================================================================
    // Initialization
    // ============================================================================

    /// Initialize mask cache: allocate `context_size / query_size` temporary buffers.
    /// Call once during setup before inference.
    /// @throws std::runtime_error if context_size not divisible by query_size
    template <typename HFADesc>
    void initialize_mask_cache(const HFADesc& hfa_desc, const std::string& device_name, AllocatorFn allocator) {
        // Get mask tensor shape from the tile model
        const size_t mask_input_idx = hfa_desc._sdpa_attention_info._tile_input_indices.mask;
        const auto& mask_port = hfa_desc._compiled_tile_model->inputs()[mask_input_idx];
        const auto mask_shape = mask_port.get_shape();
        const auto mask_dtype = mask_port.get_element_type();

        // Calculate maximum number of tiles based on context size
        const size_t context_size = hfa_desc._sdpa_attention_info._context_size;
        const size_t query_size = hfa_desc._sdpa_attention_info._query_size;

        // Validate configuration
        if (context_size % query_size != 0) {
            throw std::runtime_error("HFA: context_size (" + std::to_string(context_size) +
                                     ") must be divisible by query_size (" + std::to_string(query_size) + ")");
        }

        const size_t max_num_tiles = context_size / query_size;

        // Allocate temporary buffers for mask tile extraction
        m_mask_tile_buffers.clear();
        m_mask_tile_buffers.reserve(max_num_tiles);

        for (size_t i = 0; i < max_num_tiles; ++i) {
            auto mask_tile = allocator(mask_dtype, mask_shape, device_name);
            m_mask_tile_buffers.push_back(mask_tile);
        }
    }

    /// Reset all resources (mask cache + state buffers)
    void reset();

    // ============================================================================
    // Mask Cache Queries
    // ============================================================================

    /// Find cached mask tile, returns nullptr if not found
    ov::SoPtr<ov::ITensor> find_cached_mask_tile(const ov::SoPtr<ov::ITensor>& mask_tensor,
                                                 int64_t mask_offset,
                                                 int64_t tile_length) const;

    /// Get temporary buffer for mask extraction (throws if out of bounds)
    ov::SoPtr<ov::ITensor> get_mask_tile_buffer(size_t index) const;

    /// Number of temporary mask tile buffers
    size_t num_mask_tile_buffers() const {
        return m_mask_tile_buffers.size();
    }

    // ============================================================================
    // Mask Cache Modifications
    // ============================================================================

    /// Cache a mask tile (overwrites if key exists)
    void cache_mask_tile(const ov::SoPtr<ov::ITensor>& mask_tensor,
                         int64_t mask_offset,
                         int64_t tile_length,
                         const ov::SoPtr<ov::ITensor>& cached_tile);

    /// Clear mask cache (keeps buffers allocated)
    void clear_mask_cache();

    // ============================================================================
    // State Buffer Initialization
    // ============================================================================

    /// Initialize double-buffering: buffer[0] = provided, buffer[1] = allocated
    template <typename HFADesc>
    void initialize_state_buffers(const StateBuffers& initial_buffers,
                                  const HFADesc& hfa_desc,
                                  const std::string& device_name,
                                  AllocatorFn allocator) {
        m_state_buffers.emplace();
        auto& buffers = *m_state_buffers;
        m_current_buffer_idx = 0;

        // Buffer 0: reuse provided tensors
        buffers[0] = initial_buffers;

        // Buffer 1: allocate independent tensors
        const auto dtype = initial_buffers.acc->get_element_type();
        buffers[1].acc = allocator(dtype, initial_buffers.acc->get_shape(), device_name);
        buffers[1].max = allocator(dtype, initial_buffers.max->get_shape(), device_name);
        buffers[1].sum = allocator(dtype, initial_buffers.sum->get_shape(), device_name);
    }

    // ============================================================================
    // State Buffer Queries
    // ============================================================================

    /// Get current state buffers (throws if not initialized)
    const StateBuffers& get_current_state_buffers() const {
        if (!m_state_buffers.has_value()) {
            throw std::runtime_error("HFA: State buffers not initialized");
        }
        return (*m_state_buffers)[m_current_buffer_idx];
    }

    /// Get mutable current state buffers
    StateBuffers& get_current_state_buffers() {
        if (!m_state_buffers.has_value()) {
            throw std::runtime_error("HFA: State buffers not initialized");
        }
        return (*m_state_buffers)[m_current_buffer_idx];
    }

    // ============================================================================
    // State Buffer Modifications
    // ============================================================================

    /// Initialize state tensors: acc=0, max=-inf, sum=0 (static utility)
    static void initialize_state_tensors(ov::SoPtr<ov::ITensor>& acc,
                                         ov::SoPtr<ov::ITensor>& max,
                                         ov::SoPtr<ov::ITensor>& sum);

    /// Prepare next buffer asynchronously (call during NPU execution)
    void prepare_next_state_buffers();

    /// Switch to next buffer after inference
    void switch_buffers();

    /// Check if state buffers initialized
    bool has_state_buffers() const {
        return m_state_buffers.has_value();
    }
};

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

    std::reference_wrapper<const ov::ISyncInferRequest> _rq;

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
