// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "base_sync_infer_request.hpp"
#include "llm_compiled_model.hpp"
#include "llm_eagle3_extension.hpp"
#include "llm_infer_base_request.hpp"
#include "llm_lora_states.hpp"
#include "llm_prefix_caching.hpp"
#include "openvino/core/descriptor/output.hpp"

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

    // Create and initialize generate variant requests with memory sharing
    void create_generate_request_variants(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled_model);

    // Select appropriate generate request variant based on prompt length
    // Internally calculates expected total tokens (prompt + min_response_len) to ensure
    // sufficient capacity for both input prompt and minimum response generation
    std::shared_ptr<ov::IAsyncInferRequest> select_generate_request(int64_t prompt_length);

    void trim_kvcache_for_speculative_decoding(ov::SoPtr<ov::ITensor> position_ids);

    void infer_chunked_prefill(ov::SoPtr<ov::ITensor> input_ids,
                               ov::SoPtr<ov::ITensor> attention_mask,
                               ov::SoPtr<ov::ITensor> position_ids);

    void infer_whole_prefill(ov::SoPtr<ov::ITensor> input_ids,
                             ov::SoPtr<ov::ITensor> attention_mask,
                             ov::SoPtr<ov::ITensor> position_ids,
                             ov::SoPtr<ov::ITensor> input_token_ids);

    void infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                       ov::SoPtr<ov::ITensor> attention_mask,
                       ov::SoPtr<ov::ITensor> position_ids,
                       ov::SoPtr<ov::ITensor> input_token_ids);

    void infer_generate(ov::SoPtr<ov::ITensor> input_ids,
                        ov::SoPtr<ov::ITensor> attention_mask,
                        ov::SoPtr<ov::ITensor> position_ids,
                        ov::SoPtr<ov::ITensor> input_token_ids);

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
    // This infer request is optional, so can be null.
    std::shared_ptr<ov::IAsyncInferRequest> m_lm_head_request;
    ov::SoPtr<ov::ITensor> m_logits;

    std::unordered_map<std::string, ov::Output<const ov::Node>> m_prefill_in_ports;
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_prefill_out_ports;

    // Ports for the currently selected generate model variant (set once per conversation in
    // prepare_for_new_conversation)
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_kvcache_in_ports;
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_kvcache_out_ports;

    // Ports for all generate model variants - maps from request pointer to its input/output ports
    std::unordered_map<std::shared_ptr<ov::IAsyncInferRequest>,
                       std::unordered_map<std::string, ov::Output<const ov::Node>>>
        m_generate_variant_in_ports;
    std::unordered_map<std::shared_ptr<ov::IAsyncInferRequest>,
                       std::unordered_map<std::string, ov::Output<const ov::Node>>>
        m_generate_variant_out_ports;

    ov::Output<const ov::Node> m_lm_head_logits_port;

    // Cache past_key_values ports for efficient clearing in prepare_for_new_conversation
    std::vector<ov::Output<const ov::Node>> m_prefill_past_kv_ports;

    // NB: It can be either input_ids(LLM) or inputs_embeds(VLM)
    std::string m_input_ids_name;

    bool m_generate_initialized = false;

    bool m_first_run = true;

    int64_t m_first_position_id = 0;
    int32_t m_gemma_sliding_window_size = 0;

    uint64_t m_tokens_in_present_chunk = 0;

    // Support Eagle3 speculative decoding
    Eagle3Extension m_eagle3_ext;

    // Support LoRA
    std::vector<ov::SoPtr<ov::IVariableState>> m_variableStates;
    void init_lora_states();

    // To share kvcache between prefill and generate
    bool m_past_kv_bound = false;
    void bind_past_kv();

    std::string m_pre_alloc_device = "CPU";
    std::string init_pre_alloc_device();

    // Support prefix caching
    std::unique_ptr<PrefixCachingHelper> m_prefix_caching_helper;

    // Friend declarations for PrefixCachingHelper to access protected members
    friend class PrefixCachingHelper;
};

}  // namespace npuw
}  // namespace ov
