// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "openvino/core/except.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {

namespace function {

// HostFlashAttention structure definition
struct HostFlashAttention {
    // Tiled model for flash attention execution (regular tiles)
    std::shared_ptr<ov::Model> _tile_model;

    // Final tiled model for flash attention execution (with division and transpose)
    std::shared_ptr<ov::Model> _final_tile_model;

    // Tile configuration
    int64_t _tile_size = 1024;  // Default K/V tile size

    // Input/Output mapping for tiled execution
    // Inputs: [past_acc, past_max, past_d, k_tile, v_tile, q, mask_tile]
    // Outputs: [new_acc, new_max, new_d]

    // Total KV cache size for tiling
    int64_t _kv_cache_size = 0;

    // Validation helpers
    bool is_valid() const {
        return _tile_model != nullptr && _final_tile_model != nullptr && _tile_size > 0 && _kv_cache_size > 0;
    }

    // Factory method
    static std::optional<HostFlashAttention> from(const std::shared_ptr<ov::Model>& model);
};

}  // namespace function

namespace compiled {

// Compile-time host flash attention information
struct HostFlashAttention {
    // Models to compile (will be cleared after compilation)
    std::shared_ptr<ov::Model> _tile_model_to_compile;
    std::shared_ptr<ov::Model> _final_tile_model_to_compile;

    // Compiled tile model for NPU execution (regular tiles)
    ov::SoPtr<ov::ICompiledModel> _compiled_tile_model;

    // Compiled FINAL tile model for NPU execution (with division and transpose)
    ov::SoPtr<ov::ICompiledModel> _compiled_final_tile_model;

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
