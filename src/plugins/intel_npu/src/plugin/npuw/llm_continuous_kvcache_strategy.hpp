// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
    void on_reset() override;
    void on_prefill_chunk_begin(uint32_t current_prompts_len) override;
    void on_prefill_chunk_done(uint32_t current_prompts_len, uint32_t kv_position, bool is_last) override;
    void on_prefill_done() override;
    void on_generate_kv_init() override;
    void on_generate_step_done(uint32_t tokens_before, uint32_t tokens_after, uint32_t input_tokens_len) override;
};

}  // namespace npuw
}  // namespace ov
