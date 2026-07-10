// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "base_sync_infer_request.hpp"
#include "llm_compiled_model.hpp"
#include "llm_eagle3_extension.hpp"
#include "llm_infer_base_request.hpp"
#include "llm_kvcache_strategy.hpp"
#include "llm_lora_states.hpp"
#include "llm_prefix_caching.hpp"
#include "llm_stored_tokens_state.hpp"
#include "openvino/core/descriptor/output.hpp"
#include "perf.hpp"

namespace ov {
namespace test {
namespace npuw {
struct LLMVariantSwitchTestAccess;
}  // namespace npuw
}  // namespace test
}  // namespace ov

namespace ov {
namespace npuw {

class LLMInferRequest : public ov::npuw::LLMInferBaseRequest {
public:
    explicit LLMInferRequest(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled_model);

    void infer() override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

protected:
    virtual void prepare_for_new_conversation();
    void prepare_for_new_conversation(int64_t prompt_length);
    void apply_lora();
    void clear_chunk_prefill_kv_cache();
    void copy_kvcache();
    void update_kvcache_for(std::shared_ptr<ov::IAsyncInferRequest> request,
                            const PortsMap& in_ports,
                            const PortsMap& out_ports,
                            uint32_t num_tokens,
                            bool v_transposed) override;
    void copy_lincache(std::shared_ptr<ov::IAsyncInferRequest> from_request,
                       std::shared_ptr<ov::IAsyncInferRequest> to_request,
                       const std::unordered_map<std::string, ov::Output<const ov::Node>>& from_ports,
                       const std::unordered_map<std::string, ov::Output<const ov::Node>>& to_ports);
    // Share lincache tensors from the largest generate variant to all smaller variants so
    // that a variant switch requires no explicit lincache migration. Called once after
    // m_kvcache_strategy->on_initialize() and is strategy-independent.
    void share_lincache_across_generate_variants();
    // Select appropriate generate request variant based on prompt length
    // Internally calculates expected total tokens (prompt + min_response_len) to ensure
    // sufficient capacity for both input prompt and minimum response generation
    std::shared_ptr<ov::IAsyncInferRequest> select_generate_request(int64_t prompt_length);

    void trim_kvcache_for_speculative_decoding(ov::SoPtr<ov::ITensor> position_ids);

    // Returns the KV cache capacity (max storable tokens) of the currently active variant.
    uint32_t get_current_variant_capacity() const;

    // Switches to the next larger generate variant mid-decode when the current variant's
    // KV cache is full. Updates m_kvcache_request, ports, m_kvcache_variant_idx, and
    // delegates KV rebinding to the strategy. Returns false if already at the largest variant.
    bool try_switch_to_larger_variant();

    void infer_chunked_prefill(ov::SoPtr<ov::ITensor> input_ids,
                               ov::SoPtr<ov::ITensor> attention_mask,
                               ov::SoPtr<ov::ITensor> position_ids,
                               ov::SoPtr<ov::ITensor> per_layer_inputs,
                               ov::SoPtr<ov::ITensor> visual_pos_masks,
                               ov::SoPtr<ov::ITensor> deepstack_visual_embeds);
    PrefixCachingHelper* get_prefix_caching_helper(const ov::SoPtr<ov::ITensor>& position_ids);
    bool use_longrope_prefix_cache(const ov::SoPtr<ov::ITensor>& position_ids) const;

    void infer_whole_prefill(ov::SoPtr<ov::ITensor> input_ids,
                             ov::SoPtr<ov::ITensor> attention_mask,
                             ov::SoPtr<ov::ITensor> position_ids,
                             ov::SoPtr<ov::ITensor> token_type_ids,
                             ov::SoPtr<ov::ITensor> per_layer_inputs,
                             ov::SoPtr<ov::ITensor> visual_pos_masks,
                             ov::SoPtr<ov::ITensor> deepstack_visual_embeds);

    void infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                       ov::SoPtr<ov::ITensor> attention_mask,
                       ov::SoPtr<ov::ITensor> position_ids,
                       ov::SoPtr<ov::ITensor> token_type_ids,
                       ov::SoPtr<ov::ITensor> per_layer_inputs,
                       ov::SoPtr<ov::ITensor> visual_pos_masks,
                       ov::SoPtr<ov::ITensor> deepstack_visual_embeds);

    void infer_generate(ov::SoPtr<ov::ITensor> input_ids,
                        ov::SoPtr<ov::ITensor> attention_mask,
                        ov::SoPtr<ov::ITensor> position_ids,
                        ov::SoPtr<ov::ITensor> token_type_ids,
                        ov::SoPtr<ov::ITensor> per_layer_inputs);

    // Multiple generate inference request variants, each with a different KV cache size
    std::vector<std::shared_ptr<ov::IAsyncInferRequest>> m_generate_requests;

    // Currently selected generate inference request variant (selected in prepare_for_new_conversation based on prompt
    // length)
    std::shared_ptr<ov::IAsyncInferRequest> m_kvcache_request;

    std::shared_ptr<ov::IAsyncInferRequest> m_prefill_request;
    // Base infer request for prefill, used to update history size for dynamic context.
    // NOTE: This is just a casted pointer for convenience. In fact it points to the
    // same object as m_prefill_request.
    std::shared_ptr<ov::npuw::IBaseInferRequest> m_prefill_base_request;
    // Base infer requests for all generate variants, parallel to m_generate_requests.
    // Used to propagate dummy tensors to sub-requests on conversation reset, ensuring that
    // sub-requests also release stale block tensor refs.
    std::vector<std::shared_ptr<ov::npuw::IBaseInferRequest>> m_generate_base_requests;
    // This infer request is optional, so can be null.
    std::shared_ptr<ov::IAsyncInferRequest> m_lm_head_request;
    ov::SoPtr<ov::ITensor> m_logits;

    PortsMap m_prefill_in_ports;
    PortsMap m_prefill_out_ports;

    // Ports for the currently selected generate model variant (set once per conversation in
    // prepare_for_new_conversation)
    PortsMap m_kvcache_in_ports;
    PortsMap m_kvcache_out_ports;

    // Ports for all generate model variants - maps from request pointer to its input/output ports
    std::unordered_map<std::shared_ptr<ov::IAsyncInferRequest>, PortsMap> m_generate_variant_in_ports;
    std::unordered_map<std::shared_ptr<ov::IAsyncInferRequest>, PortsMap> m_generate_variant_out_ports;

    ov::Output<const ov::Node> m_lm_head_logits_port;
    // Input port of the lm_head submodel. Its tensor is shared with the
    // prefill/generate "output_embeds" output and holds the last valid token's
    // hidden state after each inference step. Used by get_tensor() for models
    // that expose a "hidden_states" output (see m_has_lm_head_hidden_states).
    ov::Output<const ov::Node> m_lm_head_embed_port;

    std::vector<std::string> m_kvcache_past_names;
    std::vector<std::string> m_lincache_past_names;

    // NB: It can be either input_ids(LLM) or inputs_embeds(VLM)
    std::string m_input_ids_name;

    bool m_generate_initialized = false;

    // Index into m_generate_requests / m_kvcache_sizes for the currently active variant.
    // Updated in prepare_for_new_conversation() and try_switch_to_larger_variant().
    size_t m_kvcache_variant_idx = 0;

    bool m_first_run = true;

    int64_t m_first_position_id = 0;

    uint64_t m_tokens_in_present_chunk = 0;

    // Support Eagle3 speculative decoding
    Eagle3Extension m_eagle3_ext;

    // Support reset of stored tokens to 0 from external pipeline
    ov::SoPtr<ov::npuw::StoredTokensState> m_stored_tokens_state;

    // Support LoRA
    std::vector<ov::SoPtr<ov::IVariableState>> m_variableStates;
    void init_lora_states();

    // To share kvcache between prefill and generate
    bool m_past_kv_bound = false;
    void bind_past_kv();

    std::string m_pre_alloc_device = "CPU";
    std::string init_pre_alloc_device();

    // Support prefix caching
    std::vector<std::unique_ptr<PrefixCachingHelper>> m_prefix_caching_helpers;

    // LLM-level profiling for 1st token generation analysis
    using MS = ov::npuw::perf::metric<ov::npuw::perf::MSec>;
    ov::npuw::perf::Profile<MS> m_llm_profile;

    // KV cache management strategy (set once in the constructor, valid for the object's lifetime)
    std::unique_ptr<LLMKVCacheStrategy> m_kvcache_strategy;

    // Friend declarations: strategies and PrefixCachingHelper need access to protected members
    friend class LLMContinuousKVCacheStrategy;
    friend class LLMBlockKVCacheStrategy;
    friend class PrefixCachingHelper;
    friend struct ov::test::npuw::LLMVariantSwitchTestAccess;
};

}  // namespace npuw
}  // namespace ov
