// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qwen3_asr_infer_request.hpp"

#include "../infer_request_utils.hpp"
#include "../logging.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"

namespace {
constexpr uint32_t INPUT_IDS_SEQ_LEN_DIM = 1u;
}  // namespace

ov::npuw::Qwen3ASRInferRequest::Qwen3ASRInferRequest(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled_model)
    : LLMInferRequest(compiled_model) {}

// ---------------------------------------------------------------------------
// infer(): dispatch to prefill or generate based on seq_len
// ---------------------------------------------------------------------------
void ov::npuw::Qwen3ASRInferRequest::infer() {
    const auto& inputs = get_inputs();

    auto input_ids = get_tensor(ov::npuw::util::find_port_by_name(inputs, m_input_ids_name).value());
    OPENVINO_ASSERT(ov::element::i64 == input_ids->get_element_type());

    auto enc_hs_opt = ov::npuw::util::find_port_by_name(inputs, layer_names::encoder_hidden_states);
    OPENVINO_ASSERT(enc_hs_opt.has_value(), "Qwen3-ASR model must expose 'encoder_hidden_states' input");
    auto enc_hidden_states = get_tensor(enc_hs_opt.value());

    if (input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM] != 1u ||
        m_npuw_llm_compiled_model->m_kvcache_desc.num_stored_tokens == 0u) {
        infer_prefill(input_ids, enc_hidden_states);
    } else {
        infer_generate(input_ids);
    }
}

// ---------------------------------------------------------------------------
// prepare_for_new_conversation(): reset state for a fresh audio segment
// ---------------------------------------------------------------------------
void ov::npuw::Qwen3ASRInferRequest::prepare_for_new_conversation() {
    // Base class resets: prefill input buffers, strategy on_reset(), num_stored_tokens,
    // generate variant selection.
    LLMInferRequest::prepare_for_new_conversation();
    m_generate_initialized = false;
}

// ---------------------------------------------------------------------------
// infer_prefill(): right-aligned tokens with attention_mask + position_ids injection
// ---------------------------------------------------------------------------
void ov::npuw::Qwen3ASRInferRequest::infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                                   ov::SoPtr<ov::ITensor> enc_hidden_states) {
    LOG_DEBUG("[Qwen3-ASR] Calling inference for prefill model...");
    LOG_BLOCK();

    namespace uu = ov::npuw::util;
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;

    const auto prompt_length = static_cast<uint32_t>(input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM]);
    OPENVINO_ASSERT(prompt_length <= kvcache_desc.max_prompt_size,
                    "Input prompt length (",
                    prompt_length,
                    ") exceeds NPUW_LLM_MAX_PROMPT_LEN (",
                    kvcache_desc.max_prompt_size,
                    ")");

    // Select generate variant and reset strategy state.
    LLMInferRequest::prepare_for_new_conversation(static_cast<int64_t>(prompt_length));

    // RIGHT-align tokens: standard left-padding — copy to the END of the zeroed buffer.
    {
        auto padded_input = m_prefill_request->get_tensor(m_prefill_in_ports.at(m_input_ids_name));
        const size_t elem_bytes = input_ids->get_element_type().size();
        const size_t offset = (kvcache_desc.max_prompt_size - prompt_length) * elem_bytes;
        std::copy_n(reinterpret_cast<const uint8_t*>(input_ids->data()),
                    prompt_length * elem_bytes,
                    reinterpret_cast<uint8_t*>(padded_input->data()) + offset);
    }

    // Inject encoder_hidden_states (audio features): left-aligned, zero-padded to static size.
    if (enc_hidden_states) {
        if (const auto enc_it = m_prefill_in_ports.find(layer_names::encoder_hidden_states);
            enc_it != m_prefill_in_ports.end()) {
            auto padded_enc_hs = m_prefill_request->get_tensor(enc_it->second);
            OPENVINO_ASSERT(enc_hidden_states->get_byte_size() <= padded_enc_hs->get_byte_size(),
                            "encoder_hidden_states (",
                            enc_hidden_states->get_shape(),
                            ") exceeds static size (",
                            padded_enc_hs->get_shape(),
                            ")");
            uu::fill_tensor_bytes(padded_enc_hs, 0u);
            std::copy_n(reinterpret_cast<const uint8_t*>(enc_hidden_states->data()),
                        enc_hidden_states->get_byte_size(),
                        reinterpret_cast<uint8_t*>(padded_enc_hs->data()));
        }
    }

    // Inject attention_mask: 1 (mask) for left-padding, 0 (attend) for real tokens.
    // Convention from Equal(am, 0): 0 = attend, non-zero = mask.
    if (const auto attn_it = m_prefill_in_ports.find(layer_names::attention_mask);
        attn_it != m_prefill_in_ports.end()) {
        auto attn_mask = m_prefill_request->get_tensor(attn_it->second);
        if (attn_mask->get_size() >= kvcache_desc.max_prompt_size) {
            auto* mask = attn_mask->data<int64_t>();
            const auto pad_len = static_cast<int64_t>(kvcache_desc.max_prompt_size - prompt_length);
            std::fill(mask, mask + pad_len, int64_t{1});
            std::fill(mask + pad_len, mask + kvcache_desc.max_prompt_size, int64_t{0});
        }
    }

    // Inject position_ids: 0 for left-padding, [0, 1, ..., N-1] for real tokens.
    if (const auto pos_it = m_prefill_in_ports.find(layer_names::position_ids); pos_it != m_prefill_in_ports.end()) {
        auto pos_ids = m_prefill_request->get_tensor(pos_it->second);
        if (pos_ids->get_size() >= kvcache_desc.max_prompt_size) {
            auto* pos = pos_ids->data<int64_t>();
            const auto pad_len = static_cast<int64_t>(kvcache_desc.max_prompt_size - prompt_length);
            std::fill(pos, pos + pad_len, int64_t{0});
            for (int64_t i = 0; i < static_cast<int64_t>(prompt_length); ++i)
                pos[pad_len + i] = i;
        }
    }

    m_prefill_request->infer();
    kvcache_desc.num_stored_tokens += prompt_length;

    // LM head: SliceOutEmbeds slices the last token (right-aligned → last slot = last real token).
    if (m_lm_head_request) {
        m_lm_head_request->infer();
        m_logits = m_lm_head_request->get_tensor(m_lm_head_logits_port);
    } else {
        m_logits = m_prefill_request->get_tensor(m_prefill_out_ports.at(layer_names::logits));
    }

    m_generate_initialized = false;
    LOG_DEBUG("[Qwen3-ASR] Done prefill. num_stored_tokens=" << kvcache_desc.num_stored_tokens);
}

// ---------------------------------------------------------------------------
// infer_generate(): O(1) per-step decode with static KV cache
// ---------------------------------------------------------------------------
void ov::npuw::Qwen3ASRInferRequest::infer_generate(ov::SoPtr<ov::ITensor> input_ids) {
    LOG_DEBUG("[Qwen3-ASR] Calling inference for generate model...");
    LOG_BLOCK();

    namespace uu = ov::npuw::util;
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;

    OPENVINO_ASSERT(kvcache_desc.num_stored_tokens < kvcache_desc.total_size,
                    "KV-Cache is full (",
                    kvcache_desc.num_stored_tokens,
                    " / ",
                    kvcache_desc.total_size,
                    ")");

    // ------------------------------------------------------------------
    // Initialization (first generate step only):
    //   - Copy KV from prefill model into generate model.
    //   - Init attention_mask:
    //       [0]*N      (attend to valid prefill tokens at slots 0..N-1)
    //       [1]*(cap-1-N)  (mask unused/zero-padded slots N..cap-2)
    //       [0]        (attend to slot cap-1 = current token's K slot)
    // ------------------------------------------------------------------
    if (!m_generate_initialized) {
        LOG_DEBUG("[Qwen3-ASR] First generate step: copying KV from prefill model.");
        if (kvcache_desc.num_stored_tokens > 0) {
            m_kvcache_strategy->on_generate_kv_init();  // calls copy_kvcache() (left-aligned)
        }

        uu::fill_tensor_bytes(m_kvcache_request->get_tensor(m_kvcache_in_ports.at(m_input_ids_name)), 0u);

        if (const auto attn_it = m_kvcache_in_ports.find(layer_names::attention_mask);
            attn_it != m_kvcache_in_ports.end()) {
            auto attn_mask = m_kvcache_request->get_tensor(attn_it->second);
            auto* mask = attn_mask->data<int64_t>();
            const auto cap = static_cast<int64_t>(attn_mask->get_size());

            std::fill(mask, mask + kvcache_desc.num_stored_tokens, int64_t{0});  // attend prefill
            if (kvcache_desc.num_stored_tokens < cap - 1) {
                std::fill(mask + kvcache_desc.num_stored_tokens, mask + cap - 1, int64_t{1});  // mask unused
            }
            mask[cap - 1] = int64_t{0};  // current token K always at slot cap-1
        }

        // Inject encoder_hidden_states once: the static generate model holds it for all steps.
        if (const auto enc_it = m_kvcache_in_ports.find(layer_names::encoder_hidden_states);
            enc_it != m_kvcache_in_ports.end()) {
            auto outer_enc = ov::npuw::util::find_port_by_name(get_inputs(), layer_names::encoder_hidden_states);
            if (outer_enc.has_value()) {
                auto user_enc_hs = get_tensor(outer_enc.value());
                auto kv_enc_hs = m_kvcache_request->get_tensor(enc_it->second);
                ov::npuw::util::fill_tensor_bytes(kv_enc_hs, 0u);
                OPENVINO_ASSERT(user_enc_hs->get_byte_size() <= kv_enc_hs->get_byte_size(),
                                "encoder_hidden_states exceeds static enc_pad size");
                std::copy_n(reinterpret_cast<const uint8_t*>(user_enc_hs->data()),
                            user_enc_hs->get_byte_size(),
                            reinterpret_cast<uint8_t*>(kv_enc_hs->data()));
            }
        }

        m_generate_initialized = true;
    }

    OPENVINO_ASSERT(input_ids->get_size() == 1u, "Qwen3-ASR generate expects single token input");

    // Set input_ids
    {
        auto kv_input = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(m_input_ids_name));
        std::copy_n(reinterpret_cast<const uint8_t*>(input_ids->data()),
                    input_ids->get_byte_size(),
                    reinterpret_cast<uint8_t*>(kv_input->data()));
    }

    // Set position_ids = absolute position of current token
    if (const auto pos_it = m_kvcache_in_ports.find(layer_names::position_ids); pos_it != m_kvcache_in_ports.end()) {
        auto pos_ids = m_kvcache_request->get_tensor(pos_it->second);
        OPENVINO_ASSERT(pos_ids->get_size() == 1u, "Qwen3-ASR position_ids must be [1]");
        pos_ids->data<int64_t>()[0] = static_cast<int64_t>(kvcache_desc.num_stored_tokens);
    }

    m_kvcache_request->infer();
    kvcache_desc.num_stored_tokens += 1u;

    // Kick off LM head asynchronously so KV update + mask unlock can overlap with it.
    if (m_lm_head_request) {
        m_lm_head_request->start_async();
    }

    // Persist the new token's KV outputs into the past_key_values input buffer
    // so the next generate step sees the updated context.
    if (kvcache_desc.num_stored_tokens < kvcache_desc.total_size) {
        m_kvcache_strategy->on_generate_step_done(1u);
    }

    // Unlock the newly stored KV slot by clearing its attention_mask bit.
    // on_generate_step_done wrote the new token's KV to slot (num_stored_tokens - 1).
    if (const auto attn_it = m_kvcache_in_ports.find(layer_names::attention_mask);
        attn_it != m_kvcache_in_ports.end()) {
        auto attn_mask = m_kvcache_request->get_tensor(attn_it->second);
        const auto new_slot = static_cast<int64_t>(kvcache_desc.num_stored_tokens) - 1;
        if (new_slot >= 0) {
            attn_mask->data<int64_t>()[new_slot] = int64_t{0};
        }
    }

    if (m_lm_head_request) {
        m_lm_head_request->wait();
        m_logits = m_lm_head_request->get_tensor(m_lm_head_logits_port);
    } else {
        m_logits = m_kvcache_request->get_tensor(m_kvcache_out_ports.at(layer_names::logits));
    }

    LOG_DEBUG("[Qwen3-ASR] Done. num_stored_tokens=" << kvcache_desc.num_stored_tokens);
}
