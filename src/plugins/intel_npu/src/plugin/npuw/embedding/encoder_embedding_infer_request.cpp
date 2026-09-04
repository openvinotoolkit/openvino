// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "encoder_embedding_infer_request.hpp"

#include "../infer_request_utils.hpp"
#include "../logging.hpp"
#include "../util.hpp"

ov::npuw::EncoderEmbeddingInferRequest::EncoderEmbeddingInferRequest(
    const std::shared_ptr<LLMCompiledModel>& compiled_model)
    : ov::npuw::LLMInferBaseRequest(compiled_model) {
    init_ports();

    m_prefill_request = m_npuw_llm_compiled_model->m_prefill_compiled->create_infer_request();
    for (const auto& input_port : m_prefill_request->get_compiled_model()->inputs()) {
        m_prefill_in_ports.emplace(input_port.get_any_name(), input_port);
    }

    m_prefill_out_tensor = m_prefill_request->get_tensor(m_prefill_request->get_outputs()[0]);

    // The encoder is compiled as one static forward over the whole padded sequence, so its own
    // output tensor already has the shape the caller expects and is handed back as-is. That, and
    // zeroing the padding in infer(), both rely on the hidden state being [1, L, hidden].
    const auto& out_shape = m_prefill_out_tensor->get_shape();
    const auto max_prompt_size = m_npuw_llm_compiled_model->m_kvcache_desc.max_prompt_size;
    OPENVINO_ASSERT(
        out_shape.size() == 3u && out_shape[0] == 1u && out_shape[layer_ids::INPUT_IDS_SEQ_LEN_DIM] == max_prompt_size,
        "Encoder embedding model must produce a [1, ",
        max_prompt_size,
        ", hidden] hidden state, got ",
        out_shape);
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
    // The compiled encoder is static [1, L], so reject batched inputs and mask/ids disagreements
    // up front. The raw copies below would otherwise write past the static input tensors.
    OPENVINO_ASSERT(input_ids->get_shape()[0] == 1,
                    "Encoder embedding model expects batch size 1, got shape ",
                    input_ids->get_shape());
    OPENVINO_ASSERT(attention_mask->get_shape() == input_ids->get_shape(),
                    "attention_mask shape ",
                    attention_mask->get_shape(),
                    " must match input_ids shape ",
                    input_ids->get_shape());

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

    // An encoder derives each token's position from where it sits in the sequence, so valid tokens
    // have to be at the front (right-padding): token i keeps position i, and CLS or mean pooling
    // sees the real tokens first.
    uu::fill_tensor_bytes(input_ids_in, 0u);
    uu::fill_tensor<int64_t>(attn_mask_in, 0);

    std::copy_n(reinterpret_cast<uint8_t*>(input_ids->data()),
                input_ids->get_byte_size(),
                reinterpret_cast<uint8_t*>(input_ids_in->data()));
    std::copy_n(attention_mask->data<int64_t>(), attention_mask->get_size(), attn_mask_in->data<int64_t>());

    // token_type_ids: pad with zeros and pass the user's segment ids through (right-padded, like
    // input_ids), so sentence-pair inputs keep their segment assignment. Single-segment callers
    // pass all zeros anyway.
    if (auto it = m_prefill_in_ports.find(layer_names::token_type_ids); it != m_prefill_in_ports.end()) {
        auto token_type_ids_in = m_prefill_request->get_tensor(it->second);
        uu::fill_tensor<int64_t>(token_type_ids_in, 0);
        if (auto port = uu::find_port_by_name(inputs, layer_names::token_type_ids)) {
            auto token_type_ids = get_tensor(port.value());
            OPENVINO_ASSERT(ov::element::i64 == token_type_ids->get_element_type());
            OPENVINO_ASSERT(token_type_ids->get_shape() == input_ids->get_shape(),
                            "token_type_ids shape ",
                            token_type_ids->get_shape(),
                            " must match input_ids shape ",
                            input_ids->get_shape());
            std::copy_n(token_type_ids->data<int64_t>(),
                        token_type_ids->get_size(),
                        token_type_ids_in->data<int64_t>());
        }
    }

    m_prefill_request->infer();

    // The caller reads the compiled model's output tensor directly, so only the padding needs
    // attention: zero the trailing rows so they cannot reach a consumer that pools without
    // applying attention_mask. Batch is 1, so those rows are contiguous and a byte fill is enough.
    if (prompt_len < kvcache_desc.max_prompt_size) {
        auto padding = uu::make_tensor_slice(m_prefill_out_tensor,
                                             layer_ids::INPUT_IDS_SEQ_LEN_DIM,
                                             prompt_len,
                                             kvcache_desc.max_prompt_size);
        uu::fill_tensor_bytes(padding, 0u);
    }

    LOG_DEBUG("Done");
}

ov::SoPtr<ov::ITensor> ov::npuw::EncoderEmbeddingInferRequest::get_tensor(
    const ov::Output<const ov::Node>& port) const {
    if (port == get_outputs()[0]) {
        return m_prefill_out_tensor;
    }
    return ov::ISyncInferRequest::get_tensor(port);
}
