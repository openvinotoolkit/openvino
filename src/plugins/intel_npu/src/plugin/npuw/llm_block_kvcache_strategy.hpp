// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_block_manager.hpp"
#include "llm_kvcache_strategy.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {

class LLMInferRequest;  // forward declaration — avoids circular include

/// @brief Pair of key/value block managers for one transformer layer.
struct LayerBlockManagers {
    std::unique_ptr<KVCacheBlockManager> key_manager;
    std::unique_ptr<KVCacheBlockManager> value_manager;
};

/// @brief Helper for block binding: determines whether a block uses zero-copy numbered binding
/// or copy-based tail binding, and stores the pre-parsed input ports for fast access.
struct BlockBindingHelper {
    uint32_t block_size;  ///< Tokens per block (e.g., 1024)

    std::vector<ov::Output<const ov::Node>> numbered_input_ports;  ///< block_0, block_1, …
    std::optional<ov::Output<const ov::Node>> tail_input_port;     ///< block_tail (nullopt if absent)

    static BlockBindingHelper from_ports(std::vector<ov::Output<const ov::Node>> numbered_ports,
                                         std::optional<ov::Output<const ov::Node>> tail_port,
                                         uint32_t block_size_param) {
        BlockBindingHelper helper;
        helper.block_size = block_size_param;
        helper.numbered_input_ports = std::move(numbered_ports);
        helper.tail_input_port = std::move(tail_port);
        return helper;
    }

    bool has_tail_input() const {
        return tail_input_port.has_value();
    }

    bool should_treat_as_tail(uint32_t block_idx) const {
        return has_tail_input() && block_idx >= numbered_input_ports.size();
    }

    uint32_t get_block_index_for_position(uint32_t token_position) const {
        return token_position / block_size;
    }
};

/**
 * @brief KV cache strategy for the block-based (paged) KV cache implementation.
 *
 * Owns all block KV cache logic:
 *   - a pool of fixed-size KV blocks per transformer layer
 *   - zero-copy binding of blocks directly to model inputs (prefill + generate)
 *   - fallback copy path when zero-copy is not possible
 *
 * The block pool is initialized once (on_initialize) and reset between conversations
 * (on_reset) — no per-chunk buffer copies are needed.
 */
class LLMBlockKVCacheStrategy final : public LLMKVCacheStrategy {
public:
    // PortsMap matches the alias used by LLMInferRequest for consistency.
    using PortsMap = std::unordered_map<std::string, ov::Output<const ov::Node>>;

    // Dummy tensors shared across all numbered block input ports (key + value).
    struct DummyTensors {
        ov::SoPtr<ov::ITensor> key_tensor;
        ov::SoPtr<ov::ITensor> value_tensor;
    };

    explicit LLMBlockKVCacheStrategy(LLMInferRequest& req) : LLMKVCacheStrategy(req) {}

    // -------------------------------------------------------------------------
    // LLMKVCacheStrategy interface
    // -------------------------------------------------------------------------

    void on_initialize() override;
    void on_reset() override;
    void on_prefill_chunk_begin(uint32_t current_prompts_len) override;
    void on_prefill_chunk_done(uint32_t current_prompts_len, bool is_last) override;
    void on_prefill_done() override;
    void on_generate_kv_init() override;
    void on_generate_step_done(uint32_t input_tokens_len) override;

private:
    // -------------------------------------------------------------------------
    // Private helper structs (used only during initialize())
    // -------------------------------------------------------------------------

    // Pre-computed binding helpers for one layer (key + value)
    struct LayerBlockBindingHelpers {
        BlockBindingHelper key_helper;
        BlockBindingHelper value_helper;
    };

    // Pre-computed classification for each KV output port (e.g. "present.0.key").
    struct OutputKVInfo {
        uint32_t layer_idx;
        bool is_key;
    };

    // -------------------------------------------------------------------------
    // Initialization helpers
    // -------------------------------------------------------------------------

    void create_block_managers_and_helpers(
        const PortsMap& prefill_in_ports,
        const std::vector<std::shared_ptr<ov::IAsyncInferRequest>>& generate_requests,
        const std::unordered_map<std::shared_ptr<ov::IAsyncInferRequest>, PortsMap>& gen_variant_in_ports);

    // -------------------------------------------------------------------------
    // Core KV copy/bind engine (shared by prefill and generate paths)
    // -------------------------------------------------------------------------

    void copy_outputs_to_blocks(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                const PortsMap& src_ports,
                                uint32_t num_tokens,
                                bool v_transposed,
                                uint32_t current_kv_position);

    void update_generate_bindings(uint32_t old_num_tokens,
                                  uint32_t new_num_tokens,
                                  const std::shared_ptr<ov::IAsyncInferRequest>& kvcache_request);

    // -------------------------------------------------------------------------
    // Prefill path primitives
    // -------------------------------------------------------------------------

    void load_past_kv_blocks_to_prefill(const std::shared_ptr<ov::IAsyncInferRequest>& prefill_request,
                                        const PortsMap& prefill_in_ports);

    bool redirect_prefill_outputs_to_new_blocks(const std::shared_ptr<ov::IAsyncInferRequest>& prefill_request,
                                                const PortsMap& prefill_out_ports,
                                                uint32_t num_new_tokens);

    void restore_prefill_output_buffers(const std::shared_ptr<ov::IAsyncInferRequest>& prefill_request,
                                        const PortsMap& prefill_out_ports);

    // -------------------------------------------------------------------------
    // Data members
    // -------------------------------------------------------------------------

    // Block managers: layer_idx → {key_manager, value_manager}
    std::unordered_map<uint32_t, LayerBlockManagers> m_kv_cache_block_managers;

    // Pre-computed binding helpers: variant_request → layer_idx → {key_helper, value_helper}
    std::unordered_map<std::shared_ptr<ov::IAsyncInferRequest>, std::unordered_map<uint32_t, LayerBlockBindingHelpers>>
        m_variant_block_binding_helpers;

    // output_name → {layer_idx, is_key}: pre-computed in on_initialize()
    std::unordered_map<std::string, OutputKVInfo> m_output_kv_info;

    // Snapshotted original prefill output tensors for restore_prefill_output_buffers()
    std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> m_prefill_original_output_tensors;

    // Dummy tensors shared across all numbered block input ports.
    // Stored here so on_reset() can restore all ports to dummy tensors before clear_all(),
    // releasing any live shared_ptr references held by inference requests.
    DummyTensors m_dummy_tensors;

    // Whether the most recent on_prefill_chunk_begin() chose the zero-copy path.
    bool m_zero_copy_last_chunk = false;

    // Block size in tokens — fixed at on_initialize() time, equal to m_prefill_chunk_size.
    // Cached here to avoid repeated map lookups into m_kv_cache_block_managers.
    uint32_t m_block_size = 0;
};

}  // namespace npuw
}  // namespace ov
