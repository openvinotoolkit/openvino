// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>

#include "llm_infer_base_request.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"

namespace ov {
namespace npuw {

class LLMInferRequest;  // forward declaration — strategy always outlived by its owning request

/**
 * @brief Abstract strategy interface for LLM KV cache management.
 *
 * Decouples `LLMInferRequest` from the two concrete KV cache implementations:
 *   - LLMContinuousKVCacheStrategy — single contiguous buffer, classic copy-on-chunk approach
 *   - LLMBlockKVCacheStrategy      — fixed-size block pool with zero-copy prefill optimization
 *
 * Lifecycle call sequence:
 *   1. on_initialize()                    — once after LLMInferRequest construction;
 *                                            allocates block managers, sets up buffer sharing
 *                                            across generate variants (continuous mode), etc.
 *   2. on_reset()                         — each conversation boundary
 *   3. Per prefill chunk loop:
 *      a. on_prefill_chunk_begin()        — before infer()
 *      b. infer() [called by LLMInferRequest]
 *      c. on_prefill_chunk_done()         — after infer(); is_last distinguishes the two paths:
 *           is_last=false: intermediate chunk — persist KV outputs into past buffer
 *           is_last=true:  final chunk — store outputs into blocks (block mode) or no-op
 *   4. Per generate step:
 *      a. on_generate_kv_init()           — first step only: bind/copy prefill KV into generate
 *      b. infer() [called by LLMInferRequest]
 *      c. on_generate_step_done()         — after infer(): persist new token KV, update bindings
 *      c'. on_generate_variant_switch()   — conditional, after c: if the current variant's KV
 *                                           capacity is now full, promote to the next larger one
 *                                           so the next iteration starts with it already selected.
 */
class LLMKVCacheStrategy {
public:
    /**
     * @brief Construct the strategy bound to a fully-constructed LLMInferRequest.
     *
     * The reference guarantees the request outlives the strategy (both live inside the
     * same LLMInferRequest object) and can never be null.
     */
    using PortsMap = LLMInferBaseRequest::PortsMap;

    explicit LLMKVCacheStrategy(LLMInferRequest& req) : m_req(req) {}
    virtual ~LLMKVCacheStrategy() = default;

    // One-time setup after LLMInferRequest is fully constructed.
    // May also wire KV tensor sharing across generate variants (continuous mode).
    virtual void on_initialize() = 0;

    // Conversation boundary: free/zero all KV state.
    // next_prompt_length: expected prompt length for the upcoming conversation (0 = unknown).
    // Block-based strategy uses it to decide how many block device tensors to keep warm.
    virtual void on_reset(uint32_t next_prompt_length = 0) = 0;

    // Called before each prefill chunk's infer()
    virtual void on_prefill_chunk_begin(uint32_t current_prompts_len) = 0;

    // Called after each prefill chunk's infer().
    // is_last=false: intermediate chunk — KV outputs must be persisted for the next chunk.
    // is_last=true:  final chunk — block mode stores outputs into blocks; continuous is no-op.
    virtual void on_prefill_chunk_done(uint32_t current_prompts_len, bool is_last) = 0;
    // Called once on the first generate step before infer():
    // bind/copy the accumulated prefill KV into the generate model's input ports.
    virtual void on_generate_kv_init() = 0;

    // Called after on_generate_step_done() when the active variant's KV cache is now full.
    // old_req/old_in_ports: the variant that just reached capacity (source of live KV data).
    // new_req/new_in_ports: the promoted larger variant (destination for KV migration).
    // Block-based:  rebind existing block tensors to new variant ports (zero-copy).
    // Continuous:   re-pack live tokens from old BNSD layout to new layout via a CPU temp buffer.
    virtual void on_generate_variant_switch(const std::shared_ptr<ov::IAsyncInferRequest>& old_req,
                                            const PortsMap& old_in_ports,
                                            const std::shared_ptr<ov::IAsyncInferRequest>& new_req,
                                            const PortsMap& new_in_ports) = 0;

    // Called after each generate step's infer(): persist new token KV and update bindings
    virtual void on_generate_step_done(uint32_t input_tokens_len) = 0;

    // Turn the keys already in the cache into the LongRoPE mode the next infer() will
    // rotate its queries and its new keys with, after that choice flipped mid-conversation.
    // `num_cached_tokens` rows are rewritten in place; where those rows live is what the
    // two strategies disagree on, hence the dispatch. See llm_longrope_kv.hpp.
    //
    // All or nothing: an implementation that cannot complete the turn throws having
    // written nothing, so the caller's record of which mode the cache is in stays true.
    virtual void rerotate_longrope_keys(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                        const PortsMap& in_ports,
                                        uint32_t num_cached_tokens,
                                        bool to_long) = 0;

    // Continuous prefill support.

    // Continue a prefill that keeps the first `keep` tokens and will prefill
    // `delta_len` new ones. The strategy validates every dynamic precondition
    // without touching live state, so a validation failure leaves the cache
    // exactly as it was and a corrected retry may still succeed; it then moves
    // the preserved prefix into the prefill-side layout. An exception past
    // validation leaves the cache in an unspecified state and the caller must
    // recover with reset() and a full prefill.
    virtual void continue_prefill(uint32_t keep, uint32_t delta_len) {
        (void)keep;
        (void)delta_len;
        OPENVINO_THROW("Continuous prefill is not supported by this KV cache strategy.");
    }

protected:
    LLMInferRequest& m_req;
};

}  // namespace npuw
}  // namespace ov
