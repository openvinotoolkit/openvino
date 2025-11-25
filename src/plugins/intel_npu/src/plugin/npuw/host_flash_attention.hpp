// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "openvino/core/except.hpp"
#include "openvino/openvino.hpp"

namespace ov {
namespace npuw {

namespace function {

// HostFlashAttention structure definition
struct HostFlashAttention {
    // Tiled model for flash attention execution
    std::shared_ptr<ov::Model> _tile_model;

    // Tile configuration
    int64_t _tile_size = 1024;  // Default K/V tile size

    // Input/Output mapping for tiled execution
    // Inputs: [past_acc, past_max, past_d, k_tile, v_tile, q, mask_tile]
    // Outputs: [new_acc, new_max, new_d]

    // Total KV cache size for tiling
    int64_t _kv_cache_size = 0;

    // Validation helpers
    bool is_valid() const {
        return _tile_model != nullptr && _tile_size > 0 && _kv_cache_size > 0;
    }

    // Factory method
    static std::optional<HostFlashAttention> from(const std::shared_ptr<ov::Model>& model);
};

}  // namespace function

namespace compiled {

// Compile-time host flash attention information
struct HostFlashAttention {
    // TODO: Add compile-time information

    HostFlashAttention() = default;

    // Constructor that extracts metadata
    explicit HostFlashAttention(const function::HostFlashAttention& func_hfa);
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

}  // namespace host_flash_attention
}  // namespace runtime

}  // namespace npuw
}  // namespace ov
