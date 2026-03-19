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
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

// Forward declaration
namespace ov {
namespace npuw {
class LLMCompiledModel;
}
}  // namespace ov

namespace ov {
namespace npuw {

// Helper for block binding: determines whether a block uses zero-copy numbered binding
// or copy-based tail binding, and stores the pre-parsed input ports for fast access.
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

    // Returns true when block_idx is beyond the numbered range and should use the tail port.
    // Callers may unconditionally dereference tail_input_port.value() when this returns true.
    bool should_treat_as_tail(uint32_t block_idx) const {
        return has_tail_input() && block_idx >= numbered_input_ports.size();
    }

    uint32_t get_block_index_for_position(uint32_t token_position) const {
        return token_position / block_size;
    }
};

/**
 * @brief Encapsulates all block-based KV cache logic for LLMInferRequest.
 *
 * This class is a value member of LLMInferRequest (like Eagle3Extension), keeping the
 * block-KV-cache feature cleanly decoupled from the main inference flow.
 *
 * Lifecycle:
 *   1. initialize() - detects block format, sets up managers and pre-computed helpers
 *   2. reset()      - called at the start of each new conversation
 *   3. load_past_kv_blocks_to_prefill / redirect_prefill_outputs_to_new_blocks / restore_prefill_output_buffers - per
 * prefill chunk
 *   4. init_generate_kv_block_bindings - called once on the first generate step
 *   5. update_generate_bindings - called after every generate inference
 */
class BlockKVCacheExtension {
public:
    // Reuse the PortsMap type alias from the base class for consistency
    using PortsMap = std::unordered_map<std::string, ov::Output<const ov::Node>>;

    // Pre-computed binding helpers for one layer (key + value)
    struct LayerBlockBindingHelpers {
        BlockBindingHelper key_helper;
        BlockBindingHelper value_helper;
    };

    BlockKVCacheExtension() = default;

    // -------------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------------

    /**
     * @brief Detect block-based KV cache layout and set up all managers / helpers.
     *
     * Must be called once in LLMInferRequest constructor after prefill/generate requests
     * and port maps are ready.
     *
     * @return true if block-based KV cache was detected and initialized.
     */
    bool initialize(const std::shared_ptr<ov::IAsyncInferRequest>& prefill_request,
                    const std::vector<std::shared_ptr<ov::IAsyncInferRequest>>& generate_requests,
                    const PortsMap& prefill_in_ports,
                    const PortsMap& prefill_out_ports,
                    const std::unordered_map<std::shared_ptr<ov::IAsyncInferRequest>, PortsMap>& gen_variant_in_ports,
                    const std::string& device,
                    const std::shared_ptr<LLMCompiledModel>& compiled_model);

    // -------------------------------------------------------------------------
    // State queries
    // -------------------------------------------------------------------------

    bool is_enabled() const {
        return m_enabled;
    }

    bool is_empty() const {
        return m_kv_cache_block_managers.empty();
    }

    uint32_t get_block_size() const;

    // -------------------------------------------------------------------------
    // Conversation boundary
    // -------------------------------------------------------------------------

    /**
     * @brief Clear all block managers, called at the start of each new conversation.
     */
    void reset();

    // -------------------------------------------------------------------------
    // Prefill path
    // -------------------------------------------------------------------------

    /**
     * @brief Load previously-computed KV blocks into prefill model inputs.
     *
     * Called before each prefill chunk inference so the model can read past KV cache.
     */
    void load_past_kv_blocks_to_prefill(const std::shared_ptr<ov::IAsyncInferRequest>& prefill_request,
                                        const PortsMap& prefill_in_ports);

    /**
     * @brief Redirect prefill output ports to fresh blocks before inference (zero-copy optimisation).
     *
     * Call BEFORE prefill inference when the chunk is block-aligned.
     * Model will write directly into blocks — no post-inference copy is needed.
     *
     * @param num_new_tokens Tokens that will be written by this inference pass.
     * @return true if redirect was applied (block-aligned); call restore_prefill_output_buffers()
     *         before inference when this returns false.
     */
    bool redirect_prefill_outputs_to_new_blocks(const std::shared_ptr<ov::IAsyncInferRequest>& prefill_request,
                                                const PortsMap& prefill_out_ports,
                                                uint32_t num_new_tokens);

    bool prefill_outputs_redirected() const {
        return m_prefill_outputs_redirected;
    }

    /**
     * @brief Restore original output buffers when zero-copy cannot be used.
     *
     * Must be called before non-aligned prefill chunks to prevent block corruption.
     */
    void restore_prefill_output_buffers(const std::shared_ptr<ov::IAsyncInferRequest>& prefill_request,
                                        const PortsMap& prefill_out_ports);

    // -------------------------------------------------------------------------
    // Generate path
    // -------------------------------------------------------------------------

    /**
     * @brief Perform initial block→input binding for the first generate step.
     *
     * Called once per conversation when m_generate_initialized is false.
     * Numbered blocks get zero-copy binding; tail blocks are copied in.
     *
     * @param num_stored_tokens Total tokens in KV cache at call time.
     */
    void init_generate_kv_block_bindings(const std::shared_ptr<ov::IAsyncInferRequest>& kvcache_request,
                                         const PortsMap& kvcache_in_ports,
                                         uint32_t num_stored_tokens);

    /**
     * @brief Copy prefill outputs into blocks after inference (zero-copy fallback).
     *
     * Called when the zero-copy path was not applicable — specifically when kv_position
     * is not block-aligned (kv_position % block_size != 0, e.g. not enough tokens to fill a block in the last chunk).
     * Reads present.N.key/value outputs and copies tokens into the block managers
     * starting at kv_position.
     *
     * @param kv_position  kvcache_desc.num_stored_tokens value BEFORE this chunk was inferred.
     */
    void copy_prefill_outputs_to_blocks(const std::shared_ptr<ov::IAsyncInferRequest>& prefill_request,
                                        const PortsMap& prefill_out_ports,
                                        uint32_t num_tokens,
                                        bool v_transposed,
                                        uint32_t kv_position);

    /**
     * @brief Copy generate output into the current (partial) block, then re-bind inputs.
     *
     * Combines the two post-generate steps into one call:
     *   1. copy_outputs_to_blocks() — writes the newly generated token(s) into block storage.
     *   2. update_generate_bindings() — re-binds model inputs if a block transition occurred.
     *
     * @param old_num_tokens  kvcache_desc.num_stored_tokens BEFORE this generate step.
     * @param new_num_tokens  kvcache_desc.num_stored_tokens AFTER this generate step.
     * @param input_tokens_len  Number of tokens actually generated (usually 1).
     */
    void commit_generate_kv_and_rebind(uint32_t old_num_tokens,
                                       uint32_t new_num_tokens,
                                       uint32_t input_tokens_len,
                                       const std::shared_ptr<ov::IAsyncInferRequest>& kvcache_request,
                                       const PortsMap& kvcache_out_ports,
                                       const PortsMap& kvcache_in_ports);

private:
    // -------------------------------------------------------------------------
    // Private helper structs (used only during initialize())
    // -------------------------------------------------------------------------

    struct BlockShapeInfo {
        ov::Shape key_shape;
        ov::Shape value_shape;
        ov::element::Type elem_type;
        bool found_key = false;
        bool found_value = false;
    };

    struct DummyTensors {
        ov::SoPtr<ov::ITensor> key_tensor;
        ov::SoPtr<ov::ITensor> value_tensor;
    };

    // layer_idx → {key_block_names, value_block_names}
    using LayerBlockNames = std::unordered_map<uint32_t, std::pair<std::vector<std::string>, std::vector<std::string>>>;

    // -------------------------------------------------------------------------
    // Initialization helpers (called once from initialize())
    // -------------------------------------------------------------------------

    BlockShapeInfo find_block_shapes(const PortsMap& prefill_in_ports) const;
    DummyTensors allocate_dummy_block_tensors(const BlockShapeInfo& shapes) const;
    void set_dummy_tensors_to_all_requests(
        const DummyTensors& dummies,
        const std::shared_ptr<ov::IAsyncInferRequest>& prefill_request,
        const std::vector<std::shared_ptr<ov::IAsyncInferRequest>>& generate_requests,
        const PortsMap& prefill_in_ports,
        const std::unordered_map<std::shared_ptr<ov::IAsyncInferRequest>, PortsMap>& gen_variant_in_ports);
    LayerBlockNames parse_block_inputs_structure(const PortsMap& prefill_in_ports) const;
    void create_block_managers_and_helpers(
        const LayerBlockNames& layer_blocks,
        const PortsMap& prefill_in_ports,
        const std::vector<std::shared_ptr<ov::IAsyncInferRequest>>& generate_requests,
        const std::unordered_map<std::shared_ptr<ov::IAsyncInferRequest>, PortsMap>& gen_variant_in_ports);

    // Core KV copy engine shared by both prefill and generate copy-paths.
    // Reads present.N.key/value outputs from request and writes tokens into block_managers
    // starting at current_kv_position.
    void copy_outputs_to_blocks(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                const PortsMap& src_ports,
                                uint32_t num_tokens,
                                bool v_transposed,
                                uint32_t current_kv_position);

    // Re-bind model inputs after a generate step (called by commit_generate_kv_and_rebind).
    void update_generate_bindings(uint32_t old_num_tokens,
                                  uint32_t new_num_tokens,
                                  const std::shared_ptr<ov::IAsyncInferRequest>& kvcache_request,
                                  const PortsMap& kvcache_in_ports);

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------

    bool m_enabled = false;
    std::shared_ptr<LLMCompiledModel> m_compiled_model;
    std::string m_device;

    // Block managers: layer_idx → {key_manager, value_manager}
    std::unordered_map<uint32_t, KVCacheBlockManager::LayerBlockManagers> m_kv_cache_block_managers;

    // Pre-computed binding helpers: variant_request → layer_idx → {key_helper, value_helper}
    std::unordered_map<std::shared_ptr<ov::IAsyncInferRequest>, std::unordered_map<uint32_t, LayerBlockBindingHelpers>>
        m_variant_block_binding_helpers;

    // Zero-copy prefill output state
    std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> m_prefill_original_output_tensors;
    bool m_prefill_outputs_redirected = false;
};

}  // namespace npuw
}  // namespace ov
