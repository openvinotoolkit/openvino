// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_infer_request.hpp"

#include "llm_compiled_model.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"

#include <regex>

template <typename T>
void fill_tensor(ov::SoPtr<ov::ITensor> tensor, T fill_val, size_t offset = 0u) {
    T* tensor_data = tensor->data<T>();
    std::fill(tensor_data + offset, tensor_data + tensor->get_size(), fill_val);
}

template <typename T>
void fill_tensor(ov::Tensor tensor, T fill_val, size_t offset = 0u) {
    T* tensor_data = tensor.data<T>();
    std::fill(tensor_data + offset, tensor_data + tensor.get_size(), fill_val);
}

void copy_rows(const uint8_t* src,
               const size_t num_rows,
               const size_t row_byte_size,
               const size_t stride,
               uint8_t* dst) {
    const uint8_t* src_row_p = src;
    uint8_t* dst_row_p = dst;
    for (int row = 0; row < num_rows; ++row) {
        std::copy_n(src_row_p, row_byte_size, dst_row_p);
        src_row_p += stride;
        dst_row_p += stride;
    }
}

void copy_to(ov::SoPtr<ov::ITensor> src,
             ov::SoPtr<ov::ITensor> dst,
             const size_t dim,
             const size_t src_start,
             const size_t dst_start,
             const size_t count) {
    // FIXME: Only 2 is supported now...
    OPENVINO_ASSERT(dim == 2);
    const auto* src_p = static_cast<uint8_t*>(src->data());
          auto* dst_p = static_cast<uint8_t*>(dst->data());

    const auto C  = src->get_shape()[1];
    const auto SC = src->get_strides()[1];
    const auto SH = src->get_strides()[dim];

    for (int c = 0; c < C; ++c) {
        const auto* sp = src_p + (C * SC) + (src_start * SH);
              auto* dp = dst_p + (C * SC) + (dst_start * SH);
        copy_rows(sp, count, SH, SH, dp);
    }
}

ov::Tensor make_tensor_slice(ov::Tensor tensor, size_t dim, size_t start_pos, size_t end_pos) {
    ov::Shape start_shape(std::vector<size_t>(tensor.get_shape().size(), 0u));
    start_shape[dim] = start_pos;
    ov::Shape end_shape = tensor.get_shape();
    end_shape[dim] = end_pos;
    return ov::Tensor(tensor, start_shape, end_shape);
}

ov::npuw::LLMInferRequest::LLMInferRequest(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled_model,
                                           const ov::npuw::LLMCompiledModel::KVCacheDesc& kvcache_desc)
    : ov::ISyncInferRequest(compiled_model),
      m_kvcache_desc(kvcache_desc) {
    LOG_DEBUG("ov::npuw::LLMInferRequest::LLMInferRequest()");
    LOG_BLOCK();
    m_kvcache_request = compiled_model->kvcache_compiled->create_infer_request();
    m_prefill_request = compiled_model->prefill_compiled->create_infer_request();

    for (auto input_port : m_prefill_request->get_compiled_model()->inputs()) {
        m_prefill_in_ports.emplace(input_port.get_any_name(), input_port);
    }
    for (auto output_port : m_prefill_request->get_compiled_model()->outputs()) {
        m_prefill_out_ports.emplace(output_port.get_any_name(), output_port);
    }

    for (auto input_port : m_kvcache_request->get_compiled_model()->inputs()) {
        m_kvcache_in_ports.emplace(input_port.get_any_name(), input_port);
    }
    for (auto output_port : m_kvcache_request->get_compiled_model()->outputs()) {
        m_kvcache_out_ports.emplace(output_port.get_any_name(), output_port);
    }
    LOG_DEBUG("Done");
}

void ov::npuw::LLMInferRequest::prepare_for_new_conversation() {
    // FIXME: for input_ids it must be padding from tokenizer that not available from here
    // Get it from NPUW options
    LOG_DEBUG("ov::npuw::LLMInferRequest::prepare_for_new_conversation()");
    LOG_BLOCK();

    auto prefill_compiled = m_prefill_request->get_compiled_model();
    fill_tensor<int64_t>(m_prefill_request->get_tensor(m_prefill_in_ports.at("input_ids")), 0u);
    fill_tensor<int64_t>(m_prefill_request->get_tensor(m_prefill_in_ports.at("attention_mask")), 0u);
    fill_tensor<int64_t>(m_prefill_request->get_tensor(m_prefill_in_ports.at("position_ids")), 0u);
    fill_tensor<int64_t>(m_kvcache_request->get_tensor(m_kvcache_in_ports.at("attention_mask")), 0u);
    m_kvcache_desc.num_stored_tokens = 0u;
    LOG_DEBUG("Done");
}

void ov::npuw::LLMInferRequest::infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                              ov::SoPtr<ov::ITensor> attention_mask,
                                              ov::SoPtr<ov::ITensor> position_ids) {
    LOG_DEBUG("LLMInferRequest::infer_prefill()");
    LOG_BLOCK();

    prepare_for_new_conversation();

    auto prefill_compiled = m_prefill_request->get_compiled_model();

    auto padded_input_ids = m_prefill_request->get_tensor(m_prefill_in_ports.at("input_ids"));
    const size_t offset = padded_input_ids->get_size() - input_ids->get_size();
    std::copy_n(input_ids->data<int64_t>(),
                input_ids->get_size(),
                padded_input_ids->data<int64_t>() + offset);

    auto padded_attention_mask = m_prefill_request->get_tensor(m_prefill_in_ports.at("attention_mask"));
    std::copy_n(attention_mask->data<int64_t>(),
                attention_mask->get_size(),
                padded_attention_mask->data<int64_t>() + offset);

    auto padded_position_ids = m_prefill_request->get_tensor(m_prefill_in_ports.at("position_ids"));
    std::copy_n(position_ids->data<int64_t>(),
                position_ids->get_size(),
                padded_position_ids->data<int64_t>() + offset);

    m_prefill_request->infer();
    m_kvcache_desc.num_stored_tokens += static_cast<uint32_t>(input_ids->get_size());
    m_need_copy_kvcache = true;

    m_logits = m_prefill_request->get_tensor(m_prefill_out_ports.at("logits"));

    LOG_DEBUG("Done");
}

void ov::npuw::LLMInferRequest::infer_generate(ov::SoPtr<ov::ITensor> input_ids,
                                               ov::SoPtr<ov::ITensor> attention_mask,
                                               ov::SoPtr<ov::ITensor> position_ids) {
    LOG_DEBUG("LLMInferRequest::infer_generate()");
    LOG_BLOCK();

    // NB: KV-cache is full, further generation is impossible
    if (m_kvcache_desc.num_stored_tokens == m_kvcache_desc.total_size) {
        LOG_DEBUG("KV-Cache is full");
        return;
    }

    if (m_need_copy_kvcache) {
        LOG_DEBUG("Copying kv-cache from prefill to generate model.");
        const auto kStartOutputKVCacheLayers = 1u;
        const auto& kvcache_compiled = m_kvcache_request->get_compiled_model();
        for (int i = 0; i < kvcache_compiled->outputs().size() - 1; ++i) {
            const auto& output_name = kvcache_compiled->outputs()[kStartOutputKVCacheLayers + i].get_any_name();
            auto prefill_out_tensor = ov::make_tensor(m_prefill_request->get_tensor(m_prefill_out_ports.at(output_name)));

            const auto& input_name = std::regex_replace(output_name, std::regex("present"), "past_key_values");
            auto kvcache_in_tensor = ov::make_tensor(m_kvcache_request->get_tensor(m_kvcache_in_ports.at(input_name)));
            fill_tensor<ov::float16>(kvcache_in_tensor, 0);

            auto prefill_out_slice = make_tensor_slice(
                prefill_out_tensor, m_kvcache_desc.dim,
                m_kvcache_desc.max_prompt_size - m_kvcache_desc.num_stored_tokens, m_kvcache_desc.max_prompt_size
            );

            auto kvcache_in_slice = make_tensor_slice(
                kvcache_in_tensor, m_kvcache_desc.dim, 0u, m_kvcache_desc.num_stored_tokens
            );
            prefill_out_slice.copy_to(kvcache_in_slice);
            // FIXME: copy_to doesn't work for 31 key/values layers
            //        But it might be more optimal copy solution here.
            // copy_to(prefill_out_tensor, kvcache_in_tensor, 2,
            //         m_kvcache_desc.max_prompt_size - m_kvcache_desc.num_stored_tokens,
            //         0, m_kvcache_desc.num_stored_tokens);
        }
        LOG_DEBUG("LLMInferRequest::infer_generate(): Prepare attention mask pattern.");
        auto* attention_mask_data = m_kvcache_request->get_tensor(
            m_kvcache_in_ports.at("attention_mask"))->data<int64_t>();
        attention_mask_data[m_kvcache_desc.total_size - 1] = 1;

        m_need_copy_kvcache = false;
    }

    auto kv_input_ids = m_kvcache_request->get_tensor(m_kvcache_in_ports.at("input_ids"));
    std::copy_n(input_ids->data<int64_t>(), input_ids->get_size(), kv_input_ids->data<int64_t>());

    auto kv_attn_mask = m_kvcache_request->get_tensor(m_kvcache_in_ports.at("attention_mask"));
    std::copy_n(attention_mask->data<int64_t>(), attention_mask->get_size(), kv_attn_mask->data<int64_t>());

    auto kv_pos_ids = m_kvcache_request->get_tensor(m_kvcache_in_ports.at("position_ids"));
    std::copy_n(position_ids->data<int64_t>(), position_ids->get_size(), kv_pos_ids->data<int64_t>());

    m_kvcache_request->infer();
    m_logits = m_kvcache_request->get_tensor(m_kvcache_out_ports.at("logits"));
    m_kvcache_desc.num_stored_tokens += 1;

    LOG_DEBUG("Write KV-cache for the new token to the correct input position for next iteration.");
    const auto kStartOutputKVCacheLayers = 1u;
    const auto& kvcache_compiled = m_kvcache_request->get_compiled_model();
    for (int i = 0; i < kvcache_compiled->outputs().size() - 1; ++i) {
        const auto& output_name = kvcache_compiled->outputs()[kStartOutputKVCacheLayers + i].get_any_name();
        const auto& input_name = std::regex_replace(output_name, std::regex("present"), "past_key_values");
        auto kvcache_in_tensor = ov::make_tensor(m_kvcache_request->get_tensor(m_kvcache_in_ports.at(input_name)));
        auto kvcache_in_slice = make_tensor_slice(
            kvcache_in_tensor, m_kvcache_desc.dim, m_kvcache_desc.num_stored_tokens - 1, m_kvcache_desc.num_stored_tokens
        );
        auto kvcache_out_tensor = ov::make_tensor(m_kvcache_request->get_tensor(m_kvcache_out_ports.at(output_name)));
        kvcache_out_tensor.copy_to(kvcache_in_slice);
    }
    LOG_DEBUG("Done");
}

void ov::npuw::LLMInferRequest::infer() {
    const auto& inputs = get_inputs();

    auto input_ids      = get_tensor(inputs[0]);
    auto attention_mask = get_tensor(inputs[1]);
    auto position_ids   = get_tensor(inputs[2]);

    if (input_ids->get_size() != 1) {
        infer_prefill(input_ids, attention_mask, position_ids);
    } else {
        infer_generate(input_ids, attention_mask, position_ids);
    }
}

ov::SoPtr<ov::ITensor> ov::npuw::LLMInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    // NB: If asked for logits...
    if (port == get_outputs()[0]) {
        return m_logits;
    }
    return ov::ISyncInferRequest::get_tensor(port);
}
