// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "whisper_infer_request.hpp"

#include <regex>

#include "../../../utils/include/intel_npu/utils/zero/zero_remote_tensor.hpp"
#include "../infer_request_utils.hpp"
#include "../logging.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"

namespace {

constexpr uint32_t INPUT_IDS_SEQ_LEN_DIM = 1;

}  // namespace

void ov::npuw::WhisperInferRequest::prepare_for_new_conversation() {
    ov::npuw::util::fill_tensor_bytes(m_prefill_request->get_tensor(m_prefill_in_ports.at(m_input_ids_name)), 0u);
    ov::npuw::util::fill_tensor<int64_t>(m_prefill_request->get_tensor(m_prefill_in_ports.at("attention_mask")), 0);
    ov::npuw::util::fill_tensor<int64_t>(m_kvcache_request->get_tensor(m_kvcache_in_ports.at("attention_mask")), 0);
    m_npuw_llm_compiled_model->m_kvcache_desc.num_stored_tokens = 0u;

    m_alignment_tensors.clear();
}

void ov::npuw::WhisperInferRequest::infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                                  ov::SoPtr<ov::ITensor> enc_hidden_states) {
    LOG_DEBUG("Calling inference for Whisper prefill model...");
    LOG_BLOCK();

    // The staging copies below use input_ids's raw element/byte count directly, with no
    // per-axis indexing to catch a mismatched shape: a batch other than 1 (the top-level
    // declaration's batch dimension is dynamic, so nothing earlier bounds it) would copy
    // more elements than the compiled [1, max_prompt_size] ports actually hold, overflowing
    // both padded_input and padded_attention_mask below.
    OPENVINO_ASSERT(input_ids->get_shape().size() == 2 && input_ids->get_shape()[0] == 1,
                    "Input ids must have shape [1, seq_len], got ",
                    input_ids->get_shape(),
                    ".");

    // Unlike the generic LLM infer_prefill(), nothing downstream re-checks this: the staging
    // copy below is a raw copy_n of the caller's own token count, and the "KV-cache full"
    // guard in infer_generate() only compares num_stored_tokens for EQUALITY against total_size
    // - a prompt longer than max_prompt_size would push num_stored_tokens straight past
    // total_size without ever tripping that guard.
    const auto prompt_length = input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM];
    OPENVINO_ASSERT(prompt_length <= m_npuw_llm_compiled_model->m_kvcache_desc.max_prompt_size,
                    "Input prompt (",
                    prompt_length,
                    " tokens) is longer than the configured Whisper max prompt size (",
                    m_npuw_llm_compiled_model->m_kvcache_desc.max_prompt_size,
                    ").");

    prepare_for_new_conversation();

    // NB: input_ids for whisper: [token, token, pad, pad]
    auto padded_input = m_prefill_request->get_tensor(m_prefill_in_ports.at(m_input_ids_name));
    std::copy_n(input_ids->data<int64_t>(), input_ids->get_size(), padded_input->data<int64_t>());

    auto padded_attention_mask = m_prefill_request->get_tensor(m_prefill_in_ports.at("attention_mask"));
    std::fill_n(padded_attention_mask->data<int64_t>(), input_ids->get_size(), 1u);

    auto encoder_hidden_states = m_prefill_request->get_tensor(m_prefill_in_ports.at("encoder_hidden_states"));
    auto remote_tensor = std::dynamic_pointer_cast<::intel_npu::ZeroRemoteTensor>(enc_hidden_states._ptr);
    void* enc_hidden_states_data = !remote_tensor ? enc_hidden_states->data() : remote_tensor->get_original_memory();
    // The top-level encoder_hidden_states batch is legitimately dynamic (see
    // ModelBuilder::build_whisper_decoder()), so nothing validated at import time bounds
    // the caller's actual tensor size here - check it directly against the compiled
    // destination's capacity before the raw byte copy.
    OPENVINO_ASSERT(enc_hidden_states->get_byte_size() <= encoder_hidden_states->get_byte_size(),
                    "Encoder hidden states (",
                    enc_hidden_states->get_byte_size(),
                    " bytes) do not fit the compiled prefill port's capacity (",
                    encoder_hidden_states->get_byte_size(),
                    " bytes).");
    std::copy_n(reinterpret_cast<uint8_t*>(enc_hidden_states_data),
                enc_hidden_states->get_byte_size(),
                reinterpret_cast<uint8_t*>(encoder_hidden_states->data()));

    m_prefill_request->infer();

    m_npuw_llm_compiled_model->m_kvcache_desc.num_stored_tokens +=
        static_cast<uint32_t>(input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM]);
    m_need_copy_kvcache = true;

    m_logits = ov::npuw::util::make_tensor_slice(m_prefill_request->get_tensor(m_prefill_out_ports.at("logits")),
                                                 1u,
                                                 0u,
                                                 m_npuw_llm_compiled_model->m_kvcache_desc.num_stored_tokens);

    // for word-level timestamps
    auto decomposed_sdpa_size = m_npuw_llm_compiled_model->m_decomposed_sdpa_size;
    for (size_t idx = 0; idx < decomposed_sdpa_size; idx++) {
        auto name = whisper_layer_names::qk_scores_ + std::to_string(idx);
        m_alignment_tensors.insert({name, m_prefill_request->get_tensor(m_prefill_out_ports.at(name))});
    }

    LOG_DEBUG("Done");
}

void ov::npuw::WhisperInferRequest::infer_generate(ov::SoPtr<ov::ITensor> input_ids) {
    LOG_DEBUG("Calling inference for Whisper generate model...");
    LOG_BLOCK();

    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    // NB: KV-cache is full, further generation is impossible
    if (kvcache_desc.num_stored_tokens == kvcache_desc.total_size) {
        OPENVINO_THROW("KV-Cache is full.");
    }

    if (m_need_copy_kvcache) {
        LOG_DEBUG("Copying kv-cache from prefill to generate model.");
        const std::size_t kStartOutputKVCacheLayers = 1u;
        const auto& kvcache_compiled = m_kvcache_request->get_compiled_model();
        // FIXME: Find only matching by names outputs and copy them, having previously checked that such inputs exist
        for (std::size_t i = 0; i < kvcache_compiled->outputs().size() - 1; ++i) {
            const auto& output_name = kvcache_compiled->outputs()[kStartOutputKVCacheLayers + i].get_any_name();
            auto prefill_out_tensor = m_prefill_request->get_tensor(m_prefill_out_ports.at(output_name));

            // Find self-attention KVcache tensors for Whisper, cross attn copy later
            if (output_name.find("decoder") == std::string::npos) {
                continue;
            }

            const auto& input_name = std::regex_replace(output_name, std::regex("present"), "past_key_values");
            if (m_kvcache_in_ports.find(input_name) == m_kvcache_in_ports.end()) {
                LOG_DEBUG("Input name " << input_name << " doesn't contain kv cache. Skipping.");
                continue;
            }

            auto kvcache_in_tensor = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(input_name));

            // FIXME: We don't need to fill whole tensor with 0s, but only tensor.size() - num_stored_tokens
            //        taking into account kvcache dimension.
            ov::npuw::util::fill_tensor<ov::float16>(kvcache_in_tensor, 0);

            const auto is_value_tensor = output_name.find("value") != std::string::npos;
            const auto kv_dim = [&](bool v_trans) -> uint32_t {
                return (is_value_tensor && v_trans) ? 3u : kvcache_desc.dim;
            };

            const auto& pre_kv_dim = kv_dim(kvcache_desc.v_tensors_transposed_pre);
            const auto& gen_kv_dim = kv_dim(kvcache_desc.v_tensors_transposed_gen);

            auto prefill_out_slice =
                ov::npuw::util::make_tensor_slice(prefill_out_tensor, pre_kv_dim, 0, kvcache_desc.num_stored_tokens);

            auto kvcache_in_slice =
                ov::npuw::util::make_tensor_slice(kvcache_in_tensor, gen_kv_dim, 0u, kvcache_desc.num_stored_tokens);
            ov::npuw::util::copy_tensor_by_dim(prefill_out_slice, kvcache_in_slice, pre_kv_dim, gen_kv_dim);
        }

        LOG_DEBUG("Copying cross attn key value for Whisper.");
        const auto& prefill_compiled = m_prefill_request->get_compiled_model();
        for (std::size_t i = 0; i < prefill_compiled->outputs().size() - 1; ++i) {
            const auto& output_name = prefill_compiled->outputs()[kStartOutputKVCacheLayers + i].get_any_name();
            auto prefill_out_tensor = m_prefill_request->get_tensor(m_prefill_out_ports.at(output_name));

            if (output_name.find("encoder") == std::string::npos) {
                continue;
            }

            const auto& input_name = std::regex_replace(output_name, std::regex("present"), "past_key_values");
            m_kvcache_request->set_tensor(m_kvcache_in_ports.at(input_name), prefill_out_tensor);
        }

        LOG_DEBUG("Prepare attention mask pattern.");
        // NB: Prepare attention mask to be in a format [0, 0, 0, 1, 1, 1, ..., 1, 0, 1]
        auto* attention_mask_data =
            m_kvcache_request->get_tensor(m_kvcache_in_ports.at("attention_mask"))->data<uint64_t>();
        auto attention_mask_size = m_kvcache_request->get_tensor(m_kvcache_in_ports.at("attention_mask"))->get_size();
        std::fill(attention_mask_data, attention_mask_data + kvcache_desc.num_stored_tokens, 0);
        std::fill(attention_mask_data + kvcache_desc.num_stored_tokens,
                  attention_mask_data + attention_mask_size - 2,
                  1);
        attention_mask_data[attention_mask_size - 2] = 0;
        attention_mask_data[attention_mask_size - 1] = 1;

        m_need_copy_kvcache = false;
    }

    auto kv_input_ids = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(m_input_ids_name));
    std::copy_n(input_ids->data<int64_t>(), input_ids->get_size(), kv_input_ids->data<int64_t>());

    auto kv_attn_mask = m_kvcache_request->get_tensor(m_kvcache_in_ports.at("attention_mask"));
    kv_attn_mask->data<int64_t>()[kvcache_desc.num_stored_tokens - 1] = 0;

    auto kv_cache_pos = m_kvcache_request->get_tensor(m_kvcache_in_ports.at("cache_position"));
    kv_cache_pos->data<int64_t>()[0] = kvcache_desc.num_stored_tokens;

    m_kvcache_request->infer();
    m_logits = m_kvcache_request->get_tensor(m_kvcache_out_ports.at("logits"));

    kvcache_desc.num_stored_tokens += 1;

    if (kvcache_desc.num_stored_tokens == kvcache_desc.total_size) {
        // KV-cache is full, but it's not expected, so force eos_token as WA
        std::cout << "[WARNING] Whisper model stuck (KVcache is full), not full audio chunk could be processed !"
                  << std::endl;
        float* logits_data = m_logits->data<float>();  // logits for generate model has shape [1, 1, vocab_size]
        logits_data[m_npuw_llm_compiled_model->m_eos_token_id] = std::numeric_limits<float>::infinity();
        return;
    }

    LOG_DEBUG("Write KV-cache for the new token to the correct input position for next iteration.");
    const std::size_t kStartOutputKVCacheLayers = 1u;
    const auto& kvcache_compiled = m_kvcache_request->get_compiled_model();
    // FIXME: Find only matching by names outputs and copy them, having previously checked that such inputs exist
    for (std::size_t i = 0; i < kvcache_compiled->outputs().size() - 1; ++i) {
        const auto& output_name = kvcache_compiled->outputs()[kStartOutputKVCacheLayers + i].get_any_name();
        const auto& input_name = std::regex_replace(output_name, std::regex("present"), "past_key_values");
        if (m_kvcache_in_ports.find(input_name) == m_kvcache_in_ports.end()) {
            LOG_DEBUG("Input name " << input_name << " doesn't contain kv cache. Skipping.");
            continue;
        }

        auto kvcache_in_tensor = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(input_name));
        const auto& kv_dim = (output_name.find("value") != std::string::npos && kvcache_desc.v_tensors_transposed_gen)
                                 ? 3u
                                 : kvcache_desc.dim;
        auto kvcache_in_slice = ov::npuw::util::make_tensor_slice(kvcache_in_tensor,
                                                                  kv_dim,
                                                                  kvcache_desc.num_stored_tokens - 1,
                                                                  kvcache_desc.num_stored_tokens);
        auto kvcache_out_tensor = m_kvcache_request->get_tensor(m_kvcache_out_ports.at(output_name));
        ov::npuw::util::copy_tensor_by_dim(kvcache_out_tensor, kvcache_in_slice, kv_dim, kv_dim);
    }
    LOG_DEBUG("Done");
}

void ov::npuw::WhisperInferRequest::infer() {
    const auto& inputs = get_inputs();

    auto input_ids = get_tensor(ov::npuw::util::find_port_by_name(inputs, m_input_ids_name).value());
    OPENVINO_ASSERT(ov::element::i64 == input_ids->get_element_type());

    auto encoder_hidden_states = get_tensor(ov::npuw::util::find_port_by_name(inputs, "encoder_hidden_states").value());
    OPENVINO_ASSERT(ov::element::f32 == encoder_hidden_states->get_element_type());

    // NB: Check the sequence length provided for input_ids
    // in order to distinguish prefill / generate stages.
    // It's possible that for language detection, the first prefill
    // will be for a single token. So we also run prefill if
    // m_kvcache_desc.num_stored_tokens is 0.
    if (input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM] != 1 ||
        m_npuw_llm_compiled_model->m_kvcache_desc.num_stored_tokens == 0) {
        infer_prefill(input_ids, encoder_hidden_states);
    } else {
        infer_generate(input_ids);
    }
}

// FIXME: Whisper Decompose SDPA
ov::SoPtr<ov::ITensor> ov::npuw::WhisperInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    const auto& port_names = port.get_names();

    if (port_names.count(whisper_layer_names::qk_scores) > 0) {
        for (auto name : port_names) {
            if (name.find(whisper_layer_names::qk_scores_) != std::string::npos) {
                auto alignment_tensor = m_alignment_tensors.at(name);
                if (!alignment_tensor) {
                    OPENVINO_THROW(
                        "Cross-attention qk scaled scores tensor is not available. Please run inference first.");
                }
                return alignment_tensor;
            }
        }
    }

    return ov::npuw::LLMInferRequest::get_tensor(port);
}
