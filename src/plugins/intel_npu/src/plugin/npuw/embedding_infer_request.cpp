// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_infer_request.hpp"

#include <regex>

#include "infer_request_utils.hpp"
#include "logging.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "util.hpp"

ov::SoPtr<ov::ITensor> ov::npuw::EmbeddingInferRequest::create_prefill_output_tensor() {
    const auto& out_port = m_prefill_request->get_outputs()[0];
    auto out_shape = out_port.get_shape();
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;

    auto prefill_shape = ov::Shape(out_shape);
    prefill_shape[layer_ids::INPUT_IDS_SEQ_LEN_DIM] = kvcache_desc.max_prompt_size;
    return ov::get_tensor_impl(ov::Tensor(out_port.get_element_type(), prefill_shape));
}

ov::npuw::EmbeddingInferRequest::EmbeddingInferRequest(const std::shared_ptr<LLMCompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_npuw_llm_compiled_model(compiled_model) {
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

void ov::npuw::EmbeddingInferRequest::update_kvcache_for(
    std::shared_ptr<ov::IAsyncInferRequest> request,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports,
    uint32_t num_tokens,
    bool v_transposed) {
    namespace uu = ov::npuw::util;
    LOG_DEBUG("Store computed key and values for passed number of tokens in the input kv-cache"
              " layers.");
    LOG_BLOCK();
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    auto& compiled = request->get_compiled_model();
    // FIXME: Find only matching by names outputs and copy them, having previously checked that such inputs exist
    for (std::size_t i = layer_ids::kStartOutputKVCacheLayers; i < compiled->outputs().size(); ++i) {
        const auto& output_name = compiled->outputs()[i].get_any_name();
        const auto& input_name = std::regex_replace(output_name, std::regex("present"), layer_names::past_key_values);
        if (in_ports.find(input_name) == in_ports.end()) {
            // FIXME: Totally wrong debug message. input_name is an invalid name of input layer.
            LOG_DEBUG("Input name " << input_name << " doesn't contain kv cache. Skipping.");
            continue;
        }
        auto dst_tensor = request->get_tensor(in_ports.at(input_name));
        const auto& kv_dim = (output_name.find("value") != std::string::npos && v_transposed) ? 3u : kvcache_desc.dim;
        auto dst_slice = uu::make_tensor_slice(dst_tensor,
                                               kv_dim,
                                               kvcache_desc.num_stored_tokens - num_tokens,
                                               kvcache_desc.num_stored_tokens);
        auto src_tensor = request->get_tensor(out_ports.at(output_name));

        // NOTE: Sometimes present kv layer can contain greater seq_len
        //       than was sent to be processed
        uint32_t src_seq_len = static_cast<uint32_t>(src_tensor->get_shape()[kv_dim]);
        OPENVINO_ASSERT(num_tokens <= src_seq_len);
        if (src_seq_len > num_tokens) {
            auto src_slice = uu::make_tensor_slice(src_tensor, kv_dim, src_seq_len - num_tokens, src_seq_len);
            uu::copy_tensor_by_dim(src_slice, dst_slice, kv_dim, kv_dim);
        } else {
            uu::copy_tensor_by_dim(src_tensor, dst_slice, kv_dim, kv_dim);
        }
    }
    LOG_DEBUG("Done.");
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

    if (auto type_ids_port = m_prefill_in_ports.find(layer_names::token_type_ids);
        type_ids_port != m_prefill_in_ports.end()) {
        m_type_ids_in_tensor = m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::token_type_ids));
        uu::fill_tensor_bytes(m_type_ids_in_tensor, 0);
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
    LOG_DEBUG("Calling chunked inference for prefill model.");
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

        // NB: Regular LLM uses 2D position_ids [BATCH, SEQ_LEN], Qwen2.5 VL/Omni uses 3D position_ids [3, BATCH,
        // SEQ_LEN]
        // Copy postion ids with considering the 3D position_ids
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
                                                          ov::SoPtr<ov::ITensor> attention_mask,
                                                          ov::SoPtr<ov::ITensor> token_type_ids) {
    LOG_DEBUG("Calling inference for prefill model in a single launch.");
    LOG_BLOCK();

    std::copy_n(reinterpret_cast<uint8_t*>(input_ids->data()),
                input_ids->get_byte_size(),
                reinterpret_cast<uint8_t*>(m_input_ids_in_tensor->data()) + m_input_ids_in_tensor->get_byte_size() -
                    input_ids->get_byte_size());

    std::copy_n(
        attention_mask->data<int64_t>(),
        attention_mask->get_size(),
        m_attn_mask_in_tensor->data<int64_t>() + m_attn_mask_in_tensor->get_size() - attention_mask->get_size());

    if (token_type_ids != nullptr && m_type_ids_in_tensor != nullptr) {
        util::copy_to_right(token_type_ids, m_type_ids_in_tensor);
    }

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
                                                    ov::SoPtr<ov::ITensor> attention_mask,
                                                    ov::SoPtr<ov::ITensor> token_type_ids) {
    LOG_DEBUG("Calling inference for prefill model...");
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
        infer_whole_prefill(input_ids, attention_mask, token_type_ids);
    }

    LOG_DEBUG("Done");
}

void ov::npuw::EmbeddingInferRequest::infer() {
    const auto& inputs = get_inputs();

    auto input_ids = get_tensor(ov::npuw::util::find_port_by_name(inputs, layer_names::input_ids).value());
    auto attention_mask = get_tensor(ov::npuw::util::find_port_by_name(inputs, layer_names::attention_mask).value());

    auto token_type_ids = ov::npuw::util::TensorPtr();
    if (auto type_ids_port = ov::npuw::util::find_port_by_name(inputs, layer_names::token_type_ids);
        type_ids_port.has_value()) {
        token_type_ids = get_tensor(type_ids_port.value());
    }

    OPENVINO_ASSERT(ov::element::i64 == input_ids->get_element_type());
    OPENVINO_ASSERT(ov::element::i64 == attention_mask->get_element_type());

    infer_prefill(input_ids, attention_mask, token_type_ids);
}

ov::SoPtr<ov::ITensor> ov::npuw::EmbeddingInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    if (port == get_outputs()[0]) {
        return m_prefill_output;
    }

    return ov::ISyncInferRequest::get_tensor(port);
}
