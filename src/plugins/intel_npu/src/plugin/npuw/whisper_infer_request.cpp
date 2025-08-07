// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "whisper_infer_request.hpp"

#include <regex>

#include "logging.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "util_infer_request.hpp"

#include "../../utils/include/intel_npu/utils/zero/zero_remote_tensor.hpp"

namespace {

constexpr uint32_t INPUT_IDS_SEQ_LEN_DIM = 1;

} // anonymous


void ov::npuw::WhisperInferRequest::prepare_for_new_conversation() {
    ov::npuw::util::fill_tensor_bytes(m_prefill_request->get_tensor(m_prefill_in_ports.at(m_input_ids_name)), 0u);
    ov::npuw::util::fill_tensor<int64_t>(m_prefill_request->get_tensor(m_prefill_in_ports.at("attention_mask")), 0);
    ov::npuw::util::fill_tensor<int64_t>(m_kvcache_request->get_tensor(m_kvcache_in_ports.at("attention_mask")), 0);
    m_npuw_llm_compiled_model->m_kvcache_desc.num_stored_tokens = 0u;
}

void ov::npuw::WhisperInferRequest::infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                                  ov::SoPtr<ov::ITensor> enc_hidden_states) {
    LOG_DEBUG("Calling inference for Whisper prefill model...");
    LOG_BLOCK();

    prepare_for_new_conversation();

    // NB: input_ids for whisper: [token, token, pad, pad]
    auto padded_input = m_prefill_request->get_tensor(m_prefill_in_ports.at(m_input_ids_name));
    std::copy_n(input_ids->data<int64_t>(), input_ids->get_size(), padded_input->data<int64_t>());

    auto padded_attention_mask = m_prefill_request->get_tensor(m_prefill_in_ports.at("attention_mask"));
    std::fill_n(padded_attention_mask->data<int64_t>(), input_ids->get_size(), 1u);

    auto encoder_hidden_states = m_prefill_request->get_tensor(m_prefill_in_ports.at("encoder_hidden_states"));
    void* enc_hidden_states_data;
    try {
        enc_hidden_states_data = enc_hidden_states->data();
    } catch (::ov::NotImplemented) {
        auto remoteTensor = std::dynamic_pointer_cast<::intel_npu::ZeroRemoteTensor>(enc_hidden_states._ptr);
        enc_hidden_states_data = remoteTensor->get_original_memory();
    }

    std::copy_n(
        reinterpret_cast<uint8_t*>(enc_hidden_states_data),
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

            const auto& kv_dim = (output_name.find("value") != std::string::npos && kvcache_desc.v_tensors_transposed)
                                     ? 3u
                                     : kvcache_desc.dim;

            auto prefill_out_slice = ov::npuw::util::make_tensor_slice(prefill_out_tensor,
                                                                       kv_dim,
                                                                       0,
                                                                       kvcache_desc.num_stored_tokens);

            auto kvcache_in_slice = ov::npuw::util::make_tensor_slice(kvcache_in_tensor, kv_dim, 0u, kvcache_desc.num_stored_tokens);
            ov::npuw::util::copy_tensor_by_dim(prefill_out_slice, kvcache_in_slice, kv_dim);
        }

        LOG_DEBUG("Copying cross attn key value for Whisper.");
        const auto& prefill_compiled = m_prefill_request->get_compiled_model();
        for (std::size_t i = 0; i < prefill_compiled-> outputs().size() - 1; ++i) {
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
        auto* attention_mask_data = m_kvcache_request->get_tensor(m_kvcache_in_ports.at("attention_mask"))->data<uint64_t>();
        auto attention_mask_size = m_kvcache_request->get_tensor(m_kvcache_in_ports.at("attention_mask"))->get_size();
        std::fill(attention_mask_data, attention_mask_data + kvcache_desc.num_stored_tokens, 0);
        std::fill(attention_mask_data + kvcache_desc.num_stored_tokens, attention_mask_data + attention_mask_size - 2, 1);
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
        const auto& kv_dim = (output_name.find("value") != std::string::npos && kvcache_desc.v_tensors_transposed)
                                 ? 3u
                                 : kvcache_desc.dim;
        auto kvcache_in_slice = ov::npuw::util::make_tensor_slice(kvcache_in_tensor,
                                                                  kv_dim,
                                                                  kvcache_desc.num_stored_tokens - 1,
                                                                  kvcache_desc.num_stored_tokens);
        auto kvcache_out_tensor = m_kvcache_request->get_tensor(m_kvcache_out_ports.at(output_name));
        ov::npuw::util::copy_tensor_by_dim(kvcache_out_tensor, kvcache_in_slice, kv_dim);
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
    // in order to distinguish prefill / generate stages
    if (input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM] != 1) {
        infer_prefill(input_ids, encoder_hidden_states);
    } else {
        infer_generate(input_ids);
    }
}