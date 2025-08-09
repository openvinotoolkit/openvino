// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_infer_request.hpp"

#include <regex>

#include "llm_compiled_model.hpp"
#include "logging.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "util_infer_request.hpp"
namespace {
void pad_position_ids(const ov::SoPtr<ov::ITensor>& padded_position_ids, const ov::SoPtr<ov::ITensor>& position_ids) {
    // NB: Regular LLM uses 2D position_ids [BATCH, SEQ_LEN], Qwen2.5 VL/Omni uses 3D position_ids [3, BATCH, SEQ_LEN]
    // The first dimension (3) represents the three components of position encoding: time, height, and width
    // enabling alignment across multimodal inputs like text, audio, and video
    auto padded_shape = padded_position_ids->get_shape();
    auto position_shape = position_ids->get_shape();

    OPENVINO_ASSERT(position_shape.size() <= 3);

    size_t diff_dim = 0;
    for (size_t i = 0; i < padded_shape.size(); ++i) {
        if (padded_shape[i] != position_shape[i]) {
            diff_dim = i;
            break;
        }
    }

    size_t keep_elements = padded_shape[diff_dim] - position_shape[diff_dim];

    size_t batch_size = 1;
    for (size_t i = 0; i < padded_shape.size(); ++i) {
        if (i != diff_dim) {
            batch_size *= padded_shape[i];
        }
    }

    int64_t* padded_data = padded_position_ids->data<int64_t>();
    const int64_t* position_data = position_ids->data<int64_t>();

    for (size_t batch = 0; batch < batch_size; ++batch) {
        size_t padded_offset = batch * padded_shape[diff_dim];
        size_t position_offset = batch * position_shape[diff_dim];
        std::copy_n(position_data + position_offset,
                    position_shape[diff_dim],
                    padded_data + padded_offset + keep_elements);
    }
}

constexpr uint32_t INPUT_IDS_SEQ_LEN_DIM = 1;

constexpr std::size_t kStartOutputKVCacheLayers = 1;

}  // anonymous namespace

ov::npuw::LLMInferRequest::LLMInferRequest(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_npuw_llm_compiled_model(compiled_model) {
    for (const auto& input_port : m_npuw_llm_compiled_model->inputs()) {
        init_tensor(input_port);
    }
    for (const auto& output_port : m_npuw_llm_compiled_model->outputs()) {
        init_tensor(output_port);
    }

    auto input_ids_port = ov::npuw::util::find_port_by_name(compiled_model->m_prefill_compiled->inputs(), layer_names::input_ids);
    if (input_ids_port.has_value()) {
        m_input_ids_name = layer_names::input_ids;
    } else {
        OPENVINO_ASSERT(
            ov::npuw::util::find_port_by_name(compiled_model->m_prefill_compiled->inputs(), layer_names::inputs_embeds).has_value());
        m_input_ids_name = layer_names::inputs_embeds;
    }

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

    const bool use_chunk_prefill = m_npuw_llm_compiled_model->m_use_chunk_prefill;
    if (use_chunk_prefill) {
        clear_chunk_prefill_kv_cache();
    }

    if (compiled_model->m_lm_head_compiled) {
        m_lm_head_request = compiled_model->m_lm_head_compiled->create_infer_request();
        OPENVINO_ASSERT(m_lm_head_request);
        const ov::Output<const ov::Node> lm_head_embed_port = m_lm_head_request->get_inputs()[0];
        m_lm_head_logits_port = m_lm_head_request->get_outputs()[0];
        m_prefill_request->set_tensor(m_prefill_out_ports.at(layer_names::output_embeds),
                                      m_lm_head_request->get_tensor(lm_head_embed_port));
        m_kvcache_request->set_tensor(m_kvcache_out_ports.at(layer_names::output_embeds),
                                      m_lm_head_request->get_tensor(lm_head_embed_port));
    }

    // FIXME: E-177589
    // FIXME: "fixes"/workarounds caching import on CPU (also might be related to bf16 weights).
    // Unclear how it's related. Previously fill_tensor()
    // was in copy_kvcache() call. When it was removed, it broke the import accuracy.
    bool enable_cpu_wa = false;
    const auto& kvcache_compiled = m_npuw_llm_compiled_model->m_kvcache_compiled;
    for (std::size_t idx = 0; idx < kvcache_compiled->m_compiled_submodels.size(); ++idx) {
        if (kvcache_compiled->submodel_device(idx) == "CPU") {
            enable_cpu_wa = true;
            break;
        }
    }

    if (enable_cpu_wa) {
        const auto& kvcache_compiled = m_kvcache_request->get_compiled_model();
        // FIXME: Find only matching by names outputs and copy them, having previously checked that such inputs exist
        for (std::size_t i = kStartOutputKVCacheLayers; i < kvcache_compiled->outputs().size(); ++i) {
            const auto& output_name = kvcache_compiled->outputs()[i].get_any_name();
            const auto& input_name =
                std::regex_replace(output_name, std::regex("present"), layer_names::past_key_values);
            if (m_kvcache_in_ports.find(input_name) == m_kvcache_in_ports.end()) {
                continue;
            }
            auto kvcache_in_tensor = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(input_name));
            ov::npuw::util::fill_tensor<ov::float16>(kvcache_in_tensor, 0);
        }
    }

    m_generate_initialized = false;
}

void ov::npuw::LLMInferRequest::init_tensor(const ov::Output<const ov::Node>& port) {
    ov::SoPtr<ITensor> tensor;
    tensor = ov::ISyncInferRequest::get_tensor(port);

    if (!tensor) {
        const auto& shape = port.get_partial_shape();
        const bool is_dynamic = shape.is_dynamic();
        ov::Shape tensor_shape;
        if (is_dynamic) {
            for (auto&& item : shape) {
                tensor_shape.push_back(item.is_static() ? item.get_length() : 0);
            }
        } else {
            tensor_shape = shape.to_shape();
        }

        tensor = ov::make_tensor(port.get_element_type(), tensor_shape);
        set_tensor(port, tensor);
    }
}

void ov::npuw::LLMInferRequest::prepare_for_new_conversation() {
    ov::npuw::util::fill_tensor_bytes(m_prefill_request->get_tensor(m_prefill_in_ports.at(m_input_ids_name)), 0u);
    ov::npuw::util::fill_tensor<int64_t>(m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::attention_mask)), 0);
    ov::npuw::util::fill_tensor<int64_t>(m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::position_ids)), 0);
    m_npuw_llm_compiled_model->m_kvcache_desc.num_stored_tokens = 0u;
}

void ov::npuw::LLMInferRequest::copy_kvcache() {
    LOG_DEBUG("Copying kv-cache from prefill to generate model.");
    LOG_BLOCK();
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    const auto& kvcache_compiled = m_kvcache_request->get_compiled_model();
    // FIXME: Find only matching by names outputs and copy them, having previously checked that such inputs exist
    for (std::size_t i = kStartOutputKVCacheLayers; i < kvcache_compiled->outputs().size(); ++i) {
        const auto& output_name = kvcache_compiled->outputs()[i].get_any_name();
        auto prefill_out_tensor = m_prefill_request->get_tensor(m_prefill_out_ports.at(output_name));

        const auto& input_name = std::regex_replace(output_name, std::regex("present"), layer_names::past_key_values);
        if (m_kvcache_in_ports.find(input_name) == m_kvcache_in_ports.end()) {
            // FIXME: Totally wrong debug message. input_name is an invalid name of input layer.
            LOG_DEBUG("Input name " << input_name << " doesn't contain kv cache. Skipping.");
            continue;
        }
        auto kvcache_in_tensor = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(input_name));

        const auto& kv_dim = (output_name.find("value") != std::string::npos && kvcache_desc.v_tensors_transposed)
                                 ? 3u
                                 : kvcache_desc.dim;

        const auto prefill_chunk_size = m_npuw_llm_compiled_model->m_prefill_chunk_size;
        const bool use_chunk_prefill = m_npuw_llm_compiled_model->m_use_chunk_prefill;
        if (use_chunk_prefill) {
            // The chunk prefilled KV results are divided into two parts:
            // Part 1: The KV results from loops 1 to n-1 have been copied into the 'past' KV input tensor
            // Part 2: The kv results from the last loop remain in the 'present' KV output tensor
            // The task is to copy both parts into the KV-cache input tensor for the decoding process

            // Copy part 1 KV results
            auto tokens_in_present_chunk = kvcache_desc.num_stored_tokens % prefill_chunk_size;
            tokens_in_present_chunk = tokens_in_present_chunk ? tokens_in_present_chunk : prefill_chunk_size;

            // tokens_in_past_chunks may be 0 in case short prompts are prefilled in single chunk
            auto tokens_in_past_chunks = kvcache_desc.num_stored_tokens - tokens_in_present_chunk;
            if (tokens_in_past_chunks > 0) {
                auto prefill_past_kv = m_prefill_request->get_tensor(m_prefill_in_ports.at(input_name));
                auto prefill_past_kv_chunks =
                    ov::npuw::util::make_tensor_slice(prefill_past_kv, kv_dim, 0u, static_cast<uint32_t>(tokens_in_past_chunks));

                auto kvcache_past_kv_chunks =
                    ov::npuw::util::make_tensor_slice(kvcache_in_tensor, kv_dim, 0u, static_cast<uint32_t>(tokens_in_past_chunks));

                ov::npuw::util::copy_tensor_by_dim(prefill_past_kv_chunks, kvcache_past_kv_chunks, kv_dim);
            }

            // Copy part 2 KV results
            auto prefill_present_kv_chunk =
                ov::npuw::util::make_tensor_slice(prefill_out_tensor,
                                                  kv_dim,
                                                  static_cast<uint32_t>(prefill_chunk_size - tokens_in_present_chunk),
                                                  static_cast<uint32_t>(prefill_chunk_size));

            auto kvcache_last_kv_chunk = ov::npuw::util::make_tensor_slice(kvcache_in_tensor,
                                                                           kv_dim,
                                                                           static_cast<uint32_t>(tokens_in_past_chunks),
                                                                           kvcache_desc.num_stored_tokens);

            ov::npuw::util::copy_tensor_by_dim(prefill_present_kv_chunk, kvcache_last_kv_chunk, kv_dim);
        } else {
            auto prefill_out_slice = ov::npuw::util::make_tensor_slice(prefill_out_tensor,
                                                                       kv_dim,
                                                                       kvcache_desc.max_prompt_size - kvcache_desc.num_stored_tokens,
                                                                       kvcache_desc.max_prompt_size);

            auto kvcache_in_slice = ov::npuw::util::make_tensor_slice(kvcache_in_tensor, kv_dim, 0u, kvcache_desc.num_stored_tokens);

            ov::npuw::util::copy_tensor_by_dim(prefill_out_slice, kvcache_in_slice, kv_dim);
        }
    }
    LOG_DEBUG("Done.");
}

void ov::npuw::LLMInferRequest::update_kvcache_for(
    std::shared_ptr<ov::IAsyncInferRequest> request,
    std::unordered_map<std::string, ov::Output<const ov::Node>> in_ports,
    std::unordered_map<std::string, ov::Output<const ov::Node>> out_ports,
    uint32_t num_tokens) {
    LOG_DEBUG("Store computed key and values for passed number of tokens in the input kv-cache"
              " layers.");
    LOG_BLOCK();
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    auto& compiled = request->get_compiled_model();
    // FIXME: Find only matching by names outputs and copy them, having previously checked that such inputs exist
    for (std::size_t i = kStartOutputKVCacheLayers; i < compiled->outputs().size(); ++i) {
        const auto& output_name = compiled->outputs()[i].get_any_name();
        const auto& input_name = std::regex_replace(output_name, std::regex("present"), layer_names::past_key_values);
        if (in_ports.find(input_name) == in_ports.end()) {
            // FIXME: Totally wrong debug message. input_name is an invalid name of input layer.
            LOG_DEBUG("Input name " << input_name << " doesn't contain kv cache. Skipping.");
            continue;
        }
        auto dst_tensor = request->get_tensor(in_ports.at(input_name));
        const auto& kv_dim = (output_name.find("value") != std::string::npos && kvcache_desc.v_tensors_transposed)
                                 ? 3u
                                 : kvcache_desc.dim;
        auto dst_slice = ov::npuw::util::make_tensor_slice(dst_tensor,
                                                           kv_dim,
                                                           kvcache_desc.num_stored_tokens - num_tokens,
                                                           kvcache_desc.num_stored_tokens);
        auto src_tensor = request->get_tensor(out_ports.at(output_name));
        ov::npuw::util::copy_tensor_by_dim(src_tensor, dst_slice, kv_dim);
    }
    LOG_DEBUG("Done.");
}

void ov::npuw::LLMInferRequest::clear_chunk_prefill_kv_cache() {
    const auto& prefill_compiled = m_prefill_request->get_compiled_model();

    for (std::size_t i = kStartOutputKVCacheLayers; i < prefill_compiled->outputs().size(); ++i) {
        const auto& output_name = prefill_compiled->outputs()[i].get_any_name();
        const auto& input_name = std::regex_replace(output_name, std::regex("present"), "past_key_values");
        if (m_prefill_in_ports.find(input_name) == m_prefill_in_ports.end()) {
            // FIXME: Totally wrong debug message. input_name is an invalid name of input layer.
            LOG_DEBUG("Input name " << input_name << " doesn't contain kv cache. Skipping.");
            continue;
        }

        auto chunk_prefill_kvcache_in_tensor = m_prefill_request->get_tensor(m_prefill_in_ports.at(input_name));

        ov::npuw::util::fill_tensor<ov::float16>(chunk_prefill_kvcache_in_tensor, 0);
    }
}

void ov::npuw::LLMInferRequest::infer_chunked_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                                      ov::SoPtr<ov::ITensor> attention_mask,
                                                      ov::SoPtr<ov::ITensor> position_ids) {
    LOG_DEBUG("Calling chunked inference for prefill model.");
    LOG_BLOCK();

    const auto input_prompt_len = input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM];
    const auto input_ids_elem_size = input_ids->get_element_type().size();
    auto input_ids_in_tensor = m_prefill_request->get_tensor(m_prefill_in_ports.at(m_input_ids_name));
    const int64_t chunk_prompt_len = m_npuw_llm_compiled_model->m_prefill_chunk_size;

    auto attn_mask_in_tensor = m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::attention_mask));
    auto pos_ids_in_tensor = m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::position_ids));

    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;

    int64_t remaining_prompts = input_prompt_len;
    while (remaining_prompts > 0) {
        // NB: input_ids can be either fp32(VLM) or i64(LLM)
        // The last chunk may not be completely filled if the actual length of the prompts is not evenly divisible by
        // the chunk size
        auto current_prompts_len = std::min(remaining_prompts, chunk_prompt_len);

        // Populate the attention mask for the present chunk
        // For the already processed tokens, they will be added into the attention mask after inference call
        size_t last_chunk_offset = attn_mask_in_tensor->get_size() - chunk_prompt_len;
        if (current_prompts_len < chunk_prompt_len) {
            // We will populate current_prompts_len on the right side of attention mask for the processing tokens
            // If the current prompt length is smaller than the chunk prompt length,
            // clear the last chunk of the attention mask to ensure non-relevant tokens are masked
            ov::npuw::util::fill_tensor<int64_t>(attn_mask_in_tensor, 0, last_chunk_offset);
        }
        std::copy_n(attention_mask->data<int64_t>() + kvcache_desc.num_stored_tokens,
                    current_prompts_len,
                    attn_mask_in_tensor->data<int64_t>() + attn_mask_in_tensor->get_size() - current_prompts_len);

        auto current_prefill_bytes = current_prompts_len * input_ids_elem_size;
        auto prefilled_bytes = kvcache_desc.num_stored_tokens * input_ids_elem_size;
        std::copy_n(reinterpret_cast<uint8_t*>(input_ids->data()) + prefilled_bytes,
                    current_prefill_bytes,
                    reinterpret_cast<uint8_t*>(input_ids_in_tensor->data()) + input_ids_in_tensor->get_byte_size() -
                        current_prefill_bytes);

        // NB: Regular LLM uses 2D position_ids [BATCH, SEQ_LEN], Qwen2.5 VL/Omni uses 3D position_ids [3, BATCH,
        // SEQ_LEN]
        // Copy postion ids with considering the 3D position_ids
        auto last_dim = position_ids->get_shape().size() - 1;
        auto actual_position_ids_slice =
            ov::npuw::util::make_tensor_slice(position_ids,
                                              static_cast<uint32_t>(last_dim),
                                              kvcache_desc.num_stored_tokens,
                                              kvcache_desc.num_stored_tokens + static_cast<uint32_t>(current_prompts_len));
        pad_position_ids(pos_ids_in_tensor, actual_position_ids_slice);

        m_prefill_request->infer();

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
                           static_cast<uint32_t>(current_prompts_len));

        // Update attention mask for the next iteration
        std::copy_n(attn_mask_in_tensor->data<int64_t>() + attn_mask_in_tensor->get_size() - current_prompts_len,
                    current_prompts_len,
                    attn_mask_in_tensor->data<int64_t>() + kvcache_desc.num_stored_tokens - current_prompts_len);
    }

    LOG_DEBUG("Done.");
}

void ov::npuw::LLMInferRequest::infer_whole_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                                    ov::SoPtr<ov::ITensor> attention_mask,
                                                    ov::SoPtr<ov::ITensor> position_ids) {
    LOG_DEBUG("Calling inference for prefill model in a single launch.");
    LOG_BLOCK();

    prepare_for_new_conversation();

    // NB: padded_input can be either fp32(VLM) or i64(LLM)
    auto padded_input = m_prefill_request->get_tensor(m_prefill_in_ports.at(m_input_ids_name));
    std::copy_n(
        reinterpret_cast<uint8_t*>(input_ids->data()),
        input_ids->get_byte_size(),
        reinterpret_cast<uint8_t*>(padded_input->data()) + padded_input->get_byte_size() - input_ids->get_byte_size());

    auto padded_attention_mask = m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::attention_mask));
    std::copy_n(
        attention_mask->data<int64_t>(),
        attention_mask->get_size(),
        padded_attention_mask->data<int64_t>() + padded_attention_mask->get_size() - attention_mask->get_size());

    auto padded_position_ids = m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::position_ids));
    pad_position_ids(padded_position_ids, position_ids);

    m_prefill_request->infer();
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    kvcache_desc.num_stored_tokens += static_cast<uint32_t>(input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM]);

    LOG_DEBUG("Done");
}

void ov::npuw::LLMInferRequest::infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                              ov::SoPtr<ov::ITensor> attention_mask,
                                              ov::SoPtr<ov::ITensor> position_ids) {
    LOG_DEBUG("Calling inference for prefill model...");
    LOG_BLOCK();

    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    if (input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM] > kvcache_desc.max_prompt_size) {
        OPENVINO_THROW("Input prompt is longer than configured \"NPUW_LLM_MAX_PROMPT_LEN\": ",
                       kvcache_desc.max_prompt_size,
                       ".\nPlease either setup bigger "
                       "\"NPUW_LLM_MAX_PROMPT_LEN\" or shorten the prompt.");
    }

    prepare_for_new_conversation();

    const bool use_chunk_prefill = m_npuw_llm_compiled_model->m_use_chunk_prefill;
    if (use_chunk_prefill) {
        infer_chunked_prefill(input_ids, attention_mask, position_ids);
    } else {
        infer_whole_prefill(input_ids, attention_mask, position_ids);
    }

    if (m_lm_head_request) {
        LOG_DEBUG("Calling inference for LM head model.");
        m_lm_head_request->infer();
        m_logits = m_lm_head_request->get_tensor(m_lm_head_logits_port);
    } else {
        m_logits = m_prefill_request->get_tensor(m_prefill_out_ports.at(layer_names::logits));
    }

    m_generate_initialized = false;

    LOG_DEBUG("Done");
}

void ov::npuw::LLMInferRequest::infer_generate(ov::SoPtr<ov::ITensor> input_ids,
                                               ov::SoPtr<ov::ITensor> attention_mask,
                                               ov::SoPtr<ov::ITensor> position_ids) {
    LOG_DEBUG("Calling inference for generate model...");
    LOG_BLOCK();

    if (!m_generate_initialized) {
        LOG_DEBUG("Copy kv-cache from prefill to generate model.");
        copy_kvcache();

        LOG_DEBUG("Prepare attention mask pattern.");
        auto kv_attn_mask = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(layer_names::attention_mask));
        ov::npuw::util::fill_tensor<int64_t>(kv_attn_mask, 0);
        // NOTE: Attention mask pattern for generate model requires last "1" to be in the end of the mask.
        //       We can safely set this "1" once and then copy on one "1" less in the infer_generate().
        kv_attn_mask->data<int64_t>()[m_npuw_llm_compiled_model->m_kvcache_desc.total_size - 1] = 1;

        m_generate_initialized = true;
    }

    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    // NB: KV-cache is full, further generation is impossible
    if (kvcache_desc.num_stored_tokens == kvcache_desc.total_size) {
        OPENVINO_THROW("KV-Cache is full.");
    }

    // FIXME: these tensors should be shared between the parent & child models
    auto kv_input_ids = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(m_input_ids_name));
    // NB: input_ids can be either fp32(VLM) or i64(LLM)
    std::copy_n(reinterpret_cast<uint8_t*>(input_ids->data()),
                input_ids->get_byte_size(),
                reinterpret_cast<uint8_t*>(kv_input_ids->data()));

    // NOTE: Attention mask pattern for generate model requires last "1" to be in the end of the mask.
    //       As it is already set above, here we copy on one "1" unit less.
    auto kv_attn_mask = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(layer_names::attention_mask));
    std::copy_n(attention_mask->data<int64_t>(), attention_mask->get_size() - 1, kv_attn_mask->data<int64_t>());

    auto kv_pos_ids = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(layer_names::position_ids));
    std::copy_n(position_ids->data<int64_t>(), position_ids->get_size(), kv_pos_ids->data<int64_t>());

    m_kvcache_request->infer();
    kvcache_desc.num_stored_tokens += 1;

    if (m_lm_head_request) {
        LOG_DEBUG("Calling inference for LM head model asynchronously");
        m_lm_head_request->start_async();
        if (kvcache_desc.num_stored_tokens < kvcache_desc.total_size) {
            update_kvcache_for(m_kvcache_request, m_kvcache_in_ports, m_kvcache_out_ports, 1);
        }
        m_lm_head_request->wait();
        LOG_DEBUG("Calling inference for LM head model -- done.");

        m_logits = m_lm_head_request->get_tensor(m_lm_head_logits_port);
    } else {
        if (kvcache_desc.num_stored_tokens < kvcache_desc.total_size) {
            update_kvcache_for(m_kvcache_request, m_kvcache_in_ports, m_kvcache_out_ports, 1);
        }

        m_logits = m_kvcache_request->get_tensor(m_kvcache_out_ports.at(layer_names::logits));
    }

    LOG_DEBUG("Done");
}

void ov::npuw::LLMInferRequest::infer() {
    const auto& inputs = get_inputs();

    auto input_ids = get_tensor(ov::npuw::util::find_port_by_name(inputs, m_input_ids_name).value());
    auto attention_mask = get_tensor(ov::npuw::util::find_port_by_name(inputs, layer_names::attention_mask).value());
    // FIXME: position_ids might be optional for some models!
    auto position_ids = get_tensor(ov::npuw::util::find_port_by_name(inputs, layer_names::position_ids).value());

    // NB: For VLM, the "inputs_embeds" contains float values (embeddings)
    OPENVINO_ASSERT(ov::element::f32 == input_ids->get_element_type() ||
                    ov::element::i64 == input_ids->get_element_type());
    OPENVINO_ASSERT(ov::element::i64 == attention_mask->get_element_type());
    OPENVINO_ASSERT(ov::element::i64 == position_ids->get_element_type());

    // NB: Check the sequence length provided for input_ids
    // in order to distinguish prefill / generate stages
    if (input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM] != 1) {
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
