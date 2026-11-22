// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "llm_infer_base_request.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"

namespace ov {
namespace npuw {

class LLMInferRequest;  // forward declaration — helper always outlived by its owning request

/**
 * @brief Owns the SWA window size and all cache-management logic (prefill/generate
 * updates, prefill->generate handover, cross-variant sharing, attention-mask population)
 * for Sliding Window Attention (SWA) KV cache inputs. The SWA input names themselves live
 * on LLMInferRequest::m_swa_past_names; this helper reads them via m_request.
 *
 * Lifecycle call sequence (all called unconditionally from LLMInferRequest):
 *   1. share_across_generate_variants() — once, right after m_kvcache_strategy->on_initialize()
 *   2. zero_prefill_tensors()           — each conversation reset
 *   3. Per prefill chunk, after m_kvcache_strategy->on_prefill_chunk_done() (intermediate
 *      chunks only): update_prefill(num_tokens)
 *   4. First generate step, after m_kvcache_strategy->on_generate_kv_init():
 *      copy_prefill_to_generate()
 *   5. Per generate step, after m_kvcache_strategy->on_generate_step_done():
 *      update_generate(num_tokens)
 *   6. Before every prefill/generate infer(): fill_attention_masks(...) (generate with an EAGLE tree mask:
 *      fill_tree_attention_masks(...))
 *   7. On a switch to a larger generate variant, after m_kvcache_strategy->on_generate_variant_switch():
 *      migrate_on_variant_switch(...)
 */
class SwaKVCacheHelper {
public:
    using PortsMap = LLMInferBaseRequest::PortsMap;

    // window_size == 0 means the model has no SWA layers.
    SwaKVCacheHelper(LLMInferRequest& request, uint32_t window_size);

    // Shares SWA tensors from the largest generate variant to avoid migration on promotion. Variants smaller than
    // the window keep their own (whole-past) SWA tensors, see migrate_on_variant_switch().
    void share_across_generate_variants();

    // Zeroes the prefill model's SWA past-KV inputs on a new conversation.
    void zero_prefill_tensors();

    // Copies SWA past KV from the prefill model's present output into the generate model's
    // circular past input, once at prefill->generate switchover.
    void copy_prefill_to_generate();

    // Left-aligned SWA past KV update for the prefill model.
    void update_prefill(uint32_t num_tokens);

    // Circular SWA past KV update for the generate model.
    void update_generate(uint32_t num_tokens);

    // Fills the externalized sliding-window attention mask input for `request`, and, if
    // present, overlays the vision-bidirectional mask from token type ids.
    void fill_attention_masks(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                              const PortsMap& in_ports,
                              uint32_t num_new_tokens,
                              bool use_circular_layout) const;

    // Generate step with multi-token tree verification (EAGLE tree mask input): each new token sees the ring slots
    // inside its own window (by its position id) and the new tokens the tree mask allows. Falls back to
    // fill_attention_masks() without a tree mask or 2D position ids.
    void fill_tree_attention_masks(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                   const PortsMap& in_ports,
                                   const ov::SoPtr<ov::ITensor>& position_ids,
                                   uint32_t num_new_tokens) const;

    // Moves the live SWA tokens into the new variant's ring when the two variants do not share the SWA tensors
    // (a variant smaller than the window keeps its whole past).
    void migrate_on_variant_switch(const std::shared_ptr<ov::IAsyncInferRequest>& old_req,
                                   const PortsMap& old_in_ports,
                                   const std::shared_ptr<ov::IAsyncInferRequest>& new_req,
                                   const PortsMap& new_in_ports) const;

private:
    void update_cache(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                      const PortsMap& in_ports,
                      const PortsMap& out_ports,
                      uint32_t num_tokens,
                      bool v_transposed,
                      bool use_circular_layout);

    LLMInferRequest& m_request;
    uint32_t m_window_size;
};

}  // namespace npuw
}  // namespace ov
