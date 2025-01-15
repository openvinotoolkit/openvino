// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_infer_request.hpp"

#include <regex>

#include "llm_compiled_model.hpp"
#include "logging.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"

namespace {
template <typename T>
void fill_tensor(ov::SoPtr<ov::ITensor> tensor, T fill_val, size_t offset = 0u) {
    T* tensor_data = tensor->data<T>();
    std::fill(tensor_data + offset, tensor_data + tensor->get_size(), fill_val);
}

ov::SoPtr<ov::ITensor> make_tensor_slice(ov::SoPtr<ov::ITensor> tensor,
                                         uint32_t dim,
                                         uint32_t start_pos,
                                         uint32_t end_pos) {
    ov::Shape start_shape(std::vector<size_t>(tensor->get_shape().size(), 0u));
    start_shape[dim] = start_pos;
    ov::Shape end_shape = tensor->get_shape();
    end_shape[dim] = end_pos;
    return ov::get_tensor_impl(ov::Tensor(ov::make_tensor(tensor), start_shape, end_shape));
}
}  // anonymous namespace

ov::npuw::LLMInferRequest::LLMInferRequest(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled_model,
                                           const ov::npuw::LLMCompiledModel::KVCacheDesc& kvcache_desc)
    : ov::ISyncInferRequest(compiled_model),
      m_kvcache_desc(kvcache_desc) {
    m_kvcache_request = compiled_model->m_kvcache_compiled->create_infer_request();
    m_prefill_request = compiled_model->m_prefill_compiled->create_infer_request();

    for (const auto& input_port : m_prefill_request->get_compiled_model()->inputs()) {
        m_prefill_in_ports.emplace(input_port.get_any_name(), input_port);
    }
    for (const auto& output_port : m_prefill_request->get_compiled_model()->outputs()) {
        m_prefill_out_ports.emplace(output_port.get_any_name(), output_port);
    }

    for (const auto& input_port : m_kvcache_request->get_compiled_model()->inputs()) {
        m_kvcache_in_ports.emplace(input_port.get_any_name(), input_port);
    }
    for (const auto& output_port : m_kvcache_request->get_compiled_model()->outputs()) {
        m_kvcache_out_ports.emplace(output_port.get_any_name(), output_port);
    }
}

void ov::npuw::LLMInferRequest::prepare_for_new_conversation() {
    // FIXME: for input_ids it must be padding from tokenizer that not available from here
    // Get it from NPUW options
    fill_tensor<int64_t>(m_prefill_request->get_tensor(m_prefill_in_ports.at("input_ids")), 0u);
    fill_tensor<int64_t>(m_prefill_request->get_tensor(m_prefill_in_ports.at("attention_mask")), 0u);
    fill_tensor<int64_t>(m_prefill_request->get_tensor(m_prefill_in_ports.at("position_ids")), 0u);
    fill_tensor<int64_t>(m_kvcache_request->get_tensor(m_kvcache_in_ports.at("attention_mask")), 0u);
    m_kvcache_desc.num_stored_tokens = 0u;
}

void ov::npuw::LLMInferRequest::infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                              ov::SoPtr<ov::ITensor> attention_mask,
                                              ov::SoPtr<ov::ITensor> position_ids) {
    LOG_DEBUG("Calling inference for prefill model...");
    LOG_BLOCK();

    prepare_for_new_conversation();

    auto padded_input_ids = m_prefill_request->get_tensor(m_prefill_in_ports.at("input_ids"));
    const size_t offset = padded_input_ids->get_size() - input_ids->get_size();
    std::copy_n(input_ids->data<int64_t>(), input_ids->get_size(), padded_input_ids->data<int64_t>() + offset);

    auto padded_attention_mask = m_prefill_request->get_tensor(m_prefill_in_ports.at("attention_mask"));
    std::copy_n(attention_mask->data<int64_t>(),
                attention_mask->get_size(),
                padded_attention_mask->data<int64_t>() + offset);

    auto padded_position_ids = m_prefill_request->get_tensor(m_prefill_in_ports.at("position_ids"));
    std::copy_n(position_ids->data<int64_t>(), position_ids->get_size(), padded_position_ids->data<int64_t>() + offset);

    m_prefill_request->infer();
    m_kvcache_desc.num_stored_tokens += static_cast<uint32_t>(input_ids->get_size());
    m_need_copy_kvcache = true;

    m_logits = m_prefill_request->get_tensor(m_prefill_out_ports.at("logits"));

    LOG_DEBUG("Done");
}

void ov::npuw::LLMInferRequest::infer_generate(ov::SoPtr<ov::ITensor> input_ids,
                                               ov::SoPtr<ov::ITensor> attention_mask,
                                               ov::SoPtr<ov::ITensor> position_ids) {
    LOG_DEBUG("Calling inference for generate model...");
    LOG_BLOCK();

    // NB: KV-cache is full, further generation is impossible
    if (m_kvcache_desc.num_stored_tokens == m_kvcache_desc.total_size) {
        OPENVINO_THROW("KV-Cache is full.");
    }

    if (m_need_copy_kvcache) {
        LOG_DEBUG("Copying kv-cache from prefill to generate model.");
        const std::size_t kStartOutputKVCacheLayers = 1u;
        const auto& kvcache_compiled = m_kvcache_request->get_compiled_model();
        for (std::size_t i = 0; i < kvcache_compiled->outputs().size() - 1; ++i) {
            const auto& output_name = kvcache_compiled->outputs()[kStartOutputKVCacheLayers + i].get_any_name();
            auto prefill_out_tensor = m_prefill_request->get_tensor(m_prefill_out_ports.at(output_name));

            const auto& input_name = std::regex_replace(output_name, std::regex("present"), "past_key_values");
            auto kvcache_in_tensor = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(input_name));

            // FIXME: We don't need to fill whole tensor with 0s, but only tensor.size() - num_stored_tokens
            //        taking into account kvcache dimension.
            fill_tensor<ov::float16>(kvcache_in_tensor, 0);

            auto prefill_out_slice =
                make_tensor_slice(prefill_out_tensor,
                                  m_kvcache_desc.dim,
                                  m_kvcache_desc.max_prompt_size - m_kvcache_desc.num_stored_tokens,
                                  m_kvcache_desc.max_prompt_size);

            auto kvcache_in_slice =
                make_tensor_slice(kvcache_in_tensor, m_kvcache_desc.dim, 0u, m_kvcache_desc.num_stored_tokens);

            prefill_out_slice->copy_to(kvcache_in_slice._ptr);
        }
        LOG_DEBUG("Prepare attention mask pattern.");
        auto* attention_mask_data =
            m_kvcache_request->get_tensor(m_kvcache_in_ports.at("attention_mask"))->data<int64_t>();
        attention_mask_data[m_kvcache_desc.total_size - 1] = 1;

        m_need_copy_kvcache = false;
    }

    // FIXME: these tensors should be shared between the parent & child models
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
    const std::size_t kStartOutputKVCacheLayers = 1u;
    const auto& kvcache_compiled = m_kvcache_request->get_compiled_model();
    for (std::size_t i = 0; i < kvcache_compiled->outputs().size() - 1; ++i) {
        const auto& output_name = kvcache_compiled->outputs()[kStartOutputKVCacheLayers + i].get_any_name();
        const auto& input_name = std::regex_replace(output_name, std::regex("present"), "past_key_values");
        auto kvcache_in_tensor = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(input_name));
        auto kvcache_in_slice = make_tensor_slice(kvcache_in_tensor,
                                                  m_kvcache_desc.dim,
                                                  m_kvcache_desc.num_stored_tokens - 1,
                                                  m_kvcache_desc.num_stored_tokens);
        auto kvcache_out_tensor = m_kvcache_request->get_tensor(m_kvcache_out_ports.at(output_name));
        kvcache_out_tensor->copy_to(kvcache_in_slice._ptr);
    }
    LOG_DEBUG("Done");
}

void ov::npuw::LLMInferRequest::infer() {
    const auto& inputs = get_inputs();

    auto input_ids = get_tensor(inputs[0]);
    auto attention_mask = get_tensor(inputs[1]);
    auto position_ids = get_tensor(inputs[2]);

    OPENVINO_ASSERT(ov::element::i64 == input_ids->get_element_type());
    OPENVINO_ASSERT(ov::element::i64 == attention_mask->get_element_type());
    OPENVINO_ASSERT(ov::element::i64 == position_ids->get_element_type());

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
