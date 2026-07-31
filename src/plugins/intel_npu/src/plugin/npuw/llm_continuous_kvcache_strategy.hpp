// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>

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

    // Prefix caching integration (Continuous KV mode)
    // Copies the CPU KV tensors stored in each cached KVBlock's KVData into the
    // prefill model's past_key_values input ports.
    void apply_cached_prefix_blocks(const std::vector<std::shared_ptr<KVBlock>>& blocks) override;
    // Captures the prefill output slice [out_token_offset, out_token_offset+block_size) for each KV layer
    // into a new CPU-backed KVBlock.  block_token_start is the absolute token position in the prompt.
    std::shared_ptr<KVBlock> make_prefix_block(size_t block_token_start,
                                               size_t out_token_offset,
                                               size_t block_size,
                                               const std::vector<uint64_t>& token_hashes) override;

private:
    // Pre-built mapping: prefill KV output name ("present.*") → prefill input port name
    // ("past_key_values.*").  Built once in on_initialize() and reused on every
    // apply_cached_prefix_blocks() call to avoid repeated regex_replace per layer per block.
    std::unordered_map<std::string, std::string> m_prefill_out_to_in_port_map;
};

}  // namespace npuw
}  // namespace ov
