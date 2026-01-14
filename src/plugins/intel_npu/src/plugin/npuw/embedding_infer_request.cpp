// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_infer_request.hpp"

#include "infer_request_utils.hpp"
#include "logging.hpp"

ov::SoPtr<ov::ITensor> ov::npuw::EmbeddingInferRequest::create_prefill_output_tensor() {
    const auto& out_port = m_prefill_request->get_outputs()[0];
    auto out_shape = out_port.get_shape();
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;

    auto prefill_shape = ov::Shape(out_shape);
    prefill_shape[layer_ids::INPUT_IDS_SEQ_LEN_DIM] = kvcache_desc.max_prompt_size;
    return ov::get_tensor_impl(ov::Tensor(out_port.get_element_type(), prefill_shape));
}

ov::npuw::EmbeddingInferRequest::EmbeddingInferRequest(const std::shared_ptr<LLMCompiledModel>& compiled_model)
    : ov::npuw::LLMInferBaseRequest(compiled_model) {
    init_ports();

    m_prefill_request = m_npuw_llm_compiled_model->m_prefill_compiled->create_infer_request();
    for (const auto& input_port : m_prefill_request->get_compiled_model()->inputs()) {
        m_prefill_in_ports.emplace(input_port.get_any_name(), input_port);
        // Cache past_key_values ports for efficient clearing
        if (input_port.get_any_name().find(layer_names::past_key_values) != std::string::npos) {
            m_prefill_past_kv_ports.push_back(input_port);
        }
    }
    for (const auto& output_port : m_prefill_request->get_compiled_model()->outputs()) {
        m_prefill_out_ports.emplace(output_port.get_any_name(), output_port);
    }

    m_prefill_output = create_prefill_output_tensor();
}

void ov::npuw::EmbeddingInferRequest::prepare_for_new_conversation() {
    namespace uu = ov::npuw::util;

    m_attn_mask_in_tensor = m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::attention_mask));
    m_input_ids_in_tensor = m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::input_ids));

    if (auto pos_ids_port = m_prefill_in_ports.find(layer_names::position_ids);
        pos_ids_port != m_prefill_in_ports.end()) {
        m_pos_ids_in_tensor = m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::position_ids));
        uu::fill_tensor<int64_t>(m_pos_ids_in_tensor, 0);
    }

    uu::fill_tensor_bytes(m_input_ids_in_tensor, 0u);
    uu::fill_tensor<int64_t>(m_attn_mask_in_tensor, 0);
    uu::fill_tensor_bytes(m_prefill_output, 0u);

    // Clear all past_key_values tensors
    for (const auto& port : m_prefill_past_kv_ports) {
        uu::fill_tensor_bytes(m_prefill_request->get_tensor(port), 0u);
    }

    m_npuw_llm_compiled_model->m_kvcache_desc.num_stored_tokens = 0u;
}

void ov::npuw::EmbeddingInferRequest::infer_chunked_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                                            ov::SoPtr<ov::ITensor> attention_mask) {
    LOG_DEBUG("Calling embedding chunked inference for prefill model.");
    LOG_BLOCK();

    const auto input_prompt_len = input_ids->get_shape()[layer_ids::INPUT_IDS_SEQ_LEN_DIM];
    const auto input_ids_elem_size = input_ids->get_element_type().size();
    const uint64_t chunk_prompt_len = m_npuw_llm_compiled_model->m_prefill_chunk_size;

    auto position_ids = ov::make_tensor(ov::element::i64, attention_mask->get_shape());
    auto ids_data = position_ids->data<int64_t>();
    std::iota(ids_data, ids_data + position_ids->get_size(), 0);

    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    auto output_tensor = m_prefill_request->get_tensor(m_prefill_request->get_outputs()[0]);

    uint64_t remaining_prompts = input_prompt_len;
    while (remaining_prompts > 0) {
        auto current_prompts_len = std::min(remaining_prompts, chunk_prompt_len);

        // Populate the attention mask for the present chunk
        // For the already processed tokens, they will be added into the attention mask after inference call
        if (current_prompts_len < chunk_prompt_len) {
            size_t last_chunk_offset = m_attn_mask_in_tensor->get_size() - chunk_prompt_len;

            // We will populate current_prompts_len on the right side of attention mask for the processing tokens
            // If the current prompt length is smaller than the chunk prompt length,
            // clear the last chunk of the attention mask to ensure non-relevant tokens are masked
            ov::npuw::util::fill_tensor<int64_t>(m_attn_mask_in_tensor, 0, last_chunk_offset);
        }

        std::copy_n(attention_mask->data<int64_t>() + kvcache_desc.num_stored_tokens,
                    current_prompts_len,
                    m_attn_mask_in_tensor->data<int64_t>() + m_attn_mask_in_tensor->get_size() - current_prompts_len);

        const auto current_prefill_bytes = current_prompts_len * input_ids_elem_size;
        const auto prefilled_bytes = kvcache_desc.num_stored_tokens * input_ids_elem_size;

        ov::npuw::util::fill_tensor_bytes(m_input_ids_in_tensor, 0u);
        std::copy_n(reinterpret_cast<uint8_t*>(input_ids->data()) + prefilled_bytes,
                    current_prefill_bytes,
                    reinterpret_cast<uint8_t*>(m_input_ids_in_tensor->data()) + m_input_ids_in_tensor->get_byte_size() -
                        current_prefill_bytes);

        auto last_dim = position_ids->get_shape().size() - 1;
        auto actual_position_ids_slice = ov::npuw::util::make_tensor_slice(
            position_ids,
            static_cast<uint32_t>(last_dim),
            kvcache_desc.num_stored_tokens,
            kvcache_desc.num_stored_tokens + static_cast<uint32_t>(current_prompts_len));

        auto pos_ids_slice =
            ov::npuw::util::make_tensor_slice(m_pos_ids_in_tensor,
                                              static_cast<uint32_t>(last_dim),
                                              static_cast<uint32_t>(chunk_prompt_len - current_prompts_len),
                                              static_cast<uint32_t>(chunk_prompt_len));

        // Copy with proper stride handling
        actual_position_ids_slice->copy_to(pos_ids_slice._ptr);

        m_prefill_request->infer();

        auto src = ov::npuw::util::make_tensor_slice(output_tensor,
                                                     layer_ids::INPUT_IDS_SEQ_LEN_DIM,
                                                     static_cast<uint32_t>(chunk_prompt_len - current_prompts_len),
                                                     static_cast<uint32_t>(chunk_prompt_len));

        auto dst = ov::npuw::util::make_tensor_slice(
            m_prefill_output,
            layer_ids::INPUT_IDS_SEQ_LEN_DIM,
            static_cast<uint32_t>(kvcache_desc.num_stored_tokens),
            static_cast<uint32_t>(kvcache_desc.num_stored_tokens + current_prompts_len));
        ov::npuw::util::copy_tensor_by_dim(src,
                                           dst,
                                           layer_ids::INPUT_IDS_SEQ_LEN_DIM,
                                           layer_ids::INPUT_IDS_SEQ_LEN_DIM);

        remaining_prompts -= current_prompts_len;
        kvcache_desc.num_stored_tokens += static_cast<uint32_t>(current_prompts_len);

        // Do not copy last computed chunk and preserve it in present k/v layer
        if (remaining_prompts <= 0) {
            LOG_DEBUG("All prompts have been prefilled in chunks");
            break;
        }

        // Copy calculated key/values chunk from present k/v layer to past k/v layer for storage
        update_kvcache_for(m_prefill_request,
                           m_prefill_in_ports,
                           m_prefill_out_ports,
                           static_cast<uint32_t>(current_prompts_len),
                           kvcache_desc.v_tensors_transposed_pre);

        // Update attention mask for the next iteration
        std::copy_n(m_attn_mask_in_tensor->data<int64_t>() + m_attn_mask_in_tensor->get_size() - current_prompts_len,
                    current_prompts_len,
                    m_attn_mask_in_tensor->data<int64_t>() + kvcache_desc.num_stored_tokens - current_prompts_len);
    }

    LOG_DEBUG("Done.");
}

void ov::npuw::EmbeddingInferRequest::infer_whole_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                                          ov::SoPtr<ov::ITensor> attention_mask) {
    LOG_DEBUG("Calling inference for embedding prefill model in a single launch.");
    LOG_BLOCK();

    std::copy_n(reinterpret_cast<uint8_t*>(input_ids->data()),
                input_ids->get_byte_size(),
                reinterpret_cast<uint8_t*>(m_input_ids_in_tensor->data()) + m_input_ids_in_tensor->get_byte_size() -
                    input_ids->get_byte_size());

    std::copy_n(
        attention_mask->data<int64_t>(),
        attention_mask->get_size(),
        m_attn_mask_in_tensor->data<int64_t>() + m_attn_mask_in_tensor->get_size() - attention_mask->get_size());

    m_prefill_request->infer();

    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    kvcache_desc.num_stored_tokens += static_cast<uint32_t>(input_ids->get_shape()[layer_ids::INPUT_IDS_SEQ_LEN_DIM]);

    auto output_tensor = m_prefill_request->get_tensor(m_prefill_request->get_outputs()[0]);
    auto src = ov::npuw::util::make_tensor_slice(
        output_tensor,
        layer_ids::INPUT_IDS_SEQ_LEN_DIM,
        static_cast<uint32_t>(m_attn_mask_in_tensor->get_size() - attention_mask->get_size()),
        static_cast<uint32_t>(m_attn_mask_in_tensor->get_size()));

    auto dst = ov::npuw::util::make_tensor_slice(m_prefill_output,
                                                 layer_ids::INPUT_IDS_SEQ_LEN_DIM,
                                                 0,
                                                 static_cast<uint32_t>(attention_mask->get_size()));
    ov::npuw::util::copy_tensor_by_dim(src, dst, layer_ids::INPUT_IDS_SEQ_LEN_DIM, layer_ids::INPUT_IDS_SEQ_LEN_DIM);

    LOG_DEBUG("Done");
}

void ov::npuw::EmbeddingInferRequest::infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                                    ov::SoPtr<ov::ITensor> attention_mask) {
    LOG_DEBUG("Calling inference for embedding prefill model...");
    LOG_BLOCK();

    const auto prompt_length = input_ids->get_shape()[layer_ids::INPUT_IDS_SEQ_LEN_DIM];
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    if (prompt_length > kvcache_desc.max_prompt_size) {
        OPENVINO_THROW("Input prompt is longer than configured \"NPUW_LLM_MAX_PROMPT_LEN\": ",
                       kvcache_desc.max_prompt_size,
                       ".\nPlease either setup bigger "
                       "\"NPUW_LLM_MAX_PROMPT_LEN\" or shorten the prompt.");
    }

    prepare_for_new_conversation();

    const bool use_chunk_prefill = m_npuw_llm_compiled_model->m_use_chunk_prefill;
    if (use_chunk_prefill) {
        infer_chunked_prefill(input_ids, attention_mask);
    } else {
        infer_whole_prefill(input_ids, attention_mask);
    }

    LOG_DEBUG("Done");
}

void ov::npuw::EmbeddingInferRequest::infer() {
    const auto& inputs = get_inputs();

    auto input_ids = get_tensor(ov::npuw::util::find_port_by_name(inputs, layer_names::input_ids).value());
    auto attention_mask = get_tensor(ov::npuw::util::find_port_by_name(inputs, layer_names::attention_mask).value());

    OPENVINO_ASSERT(ov::element::i64 == input_ids->get_element_type());
    OPENVINO_ASSERT(ov::element::i64 == attention_mask->get_element_type());

    infer_prefill(input_ids, attention_mask);
}

ov::SoPtr<ov::ITensor> ov::npuw::EmbeddingInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    if (port == get_outputs()[0]) {
        return m_prefill_output;
    }

    return ov::ISyncInferRequest::get_tensor(port);
}
