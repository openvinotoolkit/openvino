// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "encoder_embedding_infer_request.hpp"

#include "../infer_request_utils.hpp"
#include "../logging.hpp"
#include "../util.hpp"

ov::SoPtr<ov::ITensor> ov::npuw::EncoderEmbeddingInferRequest::create_prefill_output_tensor() {
    const auto& out_port = m_prefill_request->get_outputs()[0];
    auto out_shape = out_port.get_shape();
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;

    auto prefill_shape = ov::Shape(out_shape);
    prefill_shape[layer_ids::INPUT_IDS_SEQ_LEN_DIM] = kvcache_desc.max_prompt_size;
    return ov::get_tensor_impl(ov::Tensor(out_port.get_element_type(), prefill_shape));
}

ov::npuw::EncoderEmbeddingInferRequest::EncoderEmbeddingInferRequest(
    const std::shared_ptr<LLMCompiledModel>& compiled_model)
    : ov::npuw::LLMInferBaseRequest(compiled_model) {
    init_ports();

    m_prefill_request = m_npuw_llm_compiled_model->m_prefill_compiled->create_infer_request();
    for (const auto& input_port : m_prefill_request->get_compiled_model()->inputs()) {
        m_prefill_in_ports.emplace(input_port.get_any_name(), input_port);
    }
    for (const auto& output_port : m_prefill_request->get_compiled_model()->outputs()) {
        m_prefill_out_ports.emplace(output_port.get_any_name(), output_port);
    }

    m_prefill_output = create_prefill_output_tensor();
}

void ov::npuw::EncoderEmbeddingInferRequest::infer() {
    namespace uu = ov::npuw::util;
    LOG_DEBUG("Calling inference for encoder embedding model.");
    LOG_BLOCK();

    const auto& inputs = get_inputs();
    auto input_ids = get_tensor(uu::find_port_by_name(inputs, layer_names::input_ids).value());
    auto attention_mask = get_tensor(uu::find_port_by_name(inputs, layer_names::attention_mask).value());

    OPENVINO_ASSERT(ov::element::i64 == input_ids->get_element_type());
    OPENVINO_ASSERT(ov::element::i64 == attention_mask->get_element_type());

    const auto prompt_len = static_cast<uint32_t>(input_ids->get_shape()[layer_ids::INPUT_IDS_SEQ_LEN_DIM]);
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    if (prompt_len > kvcache_desc.max_prompt_size) {
        OPENVINO_THROW("Input prompt is longer than configured \"NPUW_LLM_MAX_PROMPT_LEN\": ",
                       kvcache_desc.max_prompt_size,
                       ".\nPlease either setup bigger \"NPUW_LLM_MAX_PROMPT_LEN\" or shorten the prompt. "
                       "Note: it must not exceed the model's max_position_embeddings.");
    }

    // Static input tensors of the compiled (reshaped to [1, L]) encoder model.
    auto input_ids_in = m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::input_ids));
    auto attn_mask_in = m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::attention_mask));

    // Bidirectional encoders use learned ABSOLUTE positions, so valid tokens must sit at the front
    // (right-padding): token i keeps position i, and CLS/mean pooling sees the real tokens first.
    uu::fill_tensor_bytes(input_ids_in, 0u);
    uu::fill_tensor<int64_t>(attn_mask_in, 0);

    std::copy_n(reinterpret_cast<uint8_t*>(input_ids->data()),
                input_ids->get_byte_size(),
                reinterpret_cast<uint8_t*>(input_ids_in->data()));
    std::copy_n(attention_mask->data<int64_t>(), attention_mask->get_size(), attn_mask_in->data<int64_t>());

    // token_type_ids (single-segment) and any other auxiliary i64 inputs default to zero.
    if (auto it = m_prefill_in_ports.find(layer_names::token_type_ids); it != m_prefill_in_ports.end()) {
        uu::fill_tensor<int64_t>(m_prefill_request->get_tensor(it->second), 0);
    }

    m_prefill_request->infer();

    // Copy the valid (front) rows of the hidden state into the right-sized output tensor; the
    // remaining padded rows stay zeroed. Downstream pooling uses attention_mask to ignore them.
    uu::fill_tensor_bytes(m_prefill_output, 0u);
    auto output_tensor = m_prefill_request->get_tensor(m_prefill_request->get_outputs()[0]);
    auto src = uu::make_tensor_slice(output_tensor, layer_ids::INPUT_IDS_SEQ_LEN_DIM, 0, prompt_len);
    auto dst = uu::make_tensor_slice(m_prefill_output, layer_ids::INPUT_IDS_SEQ_LEN_DIM, 0, prompt_len);
    uu::copy_tensor_by_dim(src, dst, layer_ids::INPUT_IDS_SEQ_LEN_DIM, layer_ids::INPUT_IDS_SEQ_LEN_DIM);

    LOG_DEBUG("Done");
}

ov::SoPtr<ov::ITensor> ov::npuw::EncoderEmbeddingInferRequest::get_tensor(
    const ov::Output<const ov::Node>& port) const {
    if (port == get_outputs()[0]) {
        return m_prefill_output;
    }
    return ov::ISyncInferRequest::get_tensor(port);
}
