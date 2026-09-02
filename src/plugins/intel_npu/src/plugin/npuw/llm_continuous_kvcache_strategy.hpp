// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "llm_kvcache_strategy.hpp"

namespace ov {
namespace npuw {

class LLMInferRequest;  // forward declaration — avoids circular include

/**
 * @brief KV cache strategy for the classic continuous-buffer implementation.
 *
 * Implements the original single contiguous buffer approach:
 *   - prefill KV is accumulated in the prefill model's past_key_values input tensors
 *     via update_kvcache_for() after each intermediate chunk
 *   - after all chunks, the full KV is copied to the generate model's input buffer
 *     via copy_kvcache()
 *   - each generate step persists the new token KV via update_kvcache_for()
 */
class LLMContinuousKVCacheStrategy final : public LLMKVCacheStrategy {
public:
    explicit LLMContinuousKVCacheStrategy(LLMInferRequest& req) : LLMKVCacheStrategy(req) {}

    void on_initialize() override;
    void on_reset(uint32_t next_prompt_length = 0) override;
    void on_prefill_chunk_begin(uint32_t current_prompts_len) override;
    void on_prefill_chunk_done(uint32_t current_prompts_len, bool is_last) override;
    void on_generate_kv_init() override;
    void on_generate_variant_switch(const std::shared_ptr<ov::IAsyncInferRequest>& old_req,
                                    const PortsMap& old_in_ports,
                                    const std::shared_ptr<ov::IAsyncInferRequest>& new_req,
                                    const PortsMap& new_in_ports) override;
    void on_generate_step_done(uint32_t input_tokens_len) override;

    // Continuous prefill. Repack the preserved KV prefix [0, keep) into the prefill
    // model's past KV layout so the chunked prefill can resume from `keep`.
    void continue_prefill(uint32_t keep, uint32_t delta_len) override;

private:
    // Static per-tensor facts, derived once in on_initialize() so the
    // continuation paths do not reclassify tensors on every call.
    struct KVPairInfo {
        std::string past;
        bool is_value = false;
    };
    std::vector<KVPairInfo> m_kv_pairs;
};

}  // namespace npuw
}  // namespace ov
