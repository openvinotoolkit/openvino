// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_infer_request.hpp"

#include <regex>

#include "../../utils/include/intel_npu/utils/zero/zero_remote_tensor.hpp"
#include "llm_compiled_model.hpp"
#include "logging.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "util_xarch.hpp"

namespace {
template <typename T>
void fill_tensor(ov::SoPtr<ov::ITensor> tensor, T fill_val, size_t offset = 0u) {
    T* tensor_data = tensor->data<T>();
    std::fill(tensor_data + offset, tensor_data + tensor->get_size(), fill_val);
}

void fill_tensor_bytes(ov::SoPtr<ov::ITensor> tensor, uint8_t fill_val) {
    auto* tensor_data = reinterpret_cast<uint8_t*>(tensor->data());
    std::fill_n(tensor_data, tensor->get_byte_size(), fill_val);
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

void copy_by_planes(ov::SoPtr<ov::ITensor> src_tensor, ov::SoPtr<ov::ITensor> dst_tensor) {
    // [1, H, S1, E] -> [1, H, S2, E]
    const int N = 0;
    const int H = 1;
    const int S = 2;
    const int E = 3;

    OPENVINO_ASSERT(src_tensor->get_shape()[N] == dst_tensor->get_shape()[N]);
    OPENVINO_ASSERT(src_tensor->get_shape()[H] == dst_tensor->get_shape()[H]);
    OPENVINO_ASSERT(src_tensor->get_shape()[E] == dst_tensor->get_shape()[E]);
    OPENVINO_ASSERT(src_tensor->get_element_type() == dst_tensor->get_element_type());
    OPENVINO_ASSERT(src_tensor->get_shape()[N] == 1u);
    OPENVINO_ASSERT(src_tensor->get_shape().size() == 4u);

    const auto* src_tensor_data = reinterpret_cast<uint8_t*>(src_tensor->data());
    auto* dst_tensor_data = reinterpret_cast<uint8_t*>(dst_tensor->data());

    const auto num_planes = src_tensor->get_shape()[H];
    const auto src_plane_stride = src_tensor->get_strides()[H];
    const auto dst_plane_stride = dst_tensor->get_strides()[H];
    const auto plane_size_in_bytes = src_tensor->get_strides()[S] * src_tensor->get_shape()[S];

    for (size_t i = 0; i < num_planes; ++i) {
        std::copy_n(src_tensor_data, plane_size_in_bytes, dst_tensor_data);
        dst_tensor_data += dst_plane_stride;
        src_tensor_data += src_plane_stride;
    }
}

void copy_columns_by_row_chunks(ov::SoPtr<ov::ITensor> src, ov::SoPtr<ov::ITensor>& dst) {
    /*
      src/dst layout: [1, heads, emb_size, seq_len]

      X[*,i] - embedding for i-th token,
      Instead of copy columns, copy rows X[i,*]

      [[X00 X01 ... X0n]      [[X00 X01 ... X0n]
       [X10 X11 ... X1n]       [X10 X11 ... X1n]
       [X20 X21 ... X2n]  ...  [X20 X21 ... X2n]
             ...                     ...
       [Xm0 Xm1 ... Xmn]]      [Xm0 Xm1 ... Xmn]]
    */

    const auto src_shape = src->get_shape();

    OPENVINO_ASSERT(src_shape.size() == 4u);
    OPENVINO_ASSERT(src_shape == dst->get_shape());
    OPENVINO_ASSERT(src->get_byte_size() == dst->get_byte_size());

    const auto& src_strides = src->get_strides();
    const auto& dst_strides = dst->get_strides();
    const auto elem_size = src->get_byte_size() / src->get_size();

    const auto C = src_shape[1];
    const auto H = src_shape[2];
    const auto W = src_shape[3];

    const auto IS_H = src_strides[2];
    const auto OS_H = dst_strides[2];

    const size_t chunk_byte_size = W * elem_size;

    const auto* src_p = static_cast<uint8_t*>(src->data());
    auto* dst_p = static_cast<uint8_t*>(dst->data());

    for (size_t i = 0; i < C * H; ++i) {
        const size_t src_offset = i * IS_H;
        const size_t dst_offset = i * OS_H;
        std::copy_n(src_p + src_offset, chunk_byte_size, dst_p + dst_offset);
    }
}

std::optional<ov::Output<const ov::Node>> find_port_by_name(const std::vector<ov::Output<const ov::Node>>& ports,
                                                            const std::string& name) {
    auto it = std::find_if(ports.begin(), ports.end(), [&](const auto& port) {
        return port.get_names().count(name) != 0;
    });
    if (it == ports.end()) {
        return std::nullopt;
    }
    return std::make_optional(*it);
}

constexpr uint32_t INPUT_IDS_SEQ_LEN_DIM = 1;

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

    auto input_ids_port = find_port_by_name(compiled_model->m_prefill_compiled->inputs(), "input_ids");
    if (input_ids_port.has_value()) {
        m_input_ids_name = "input_ids";
    } else {
        OPENVINO_ASSERT(find_port_by_name(compiled_model->m_prefill_compiled->inputs(), "inputs_embeds").has_value());
        m_input_ids_name = "inputs_embeds";
    }

    auto enc_hidden_state_port =
        find_port_by_name(compiled_model->m_prefill_compiled->inputs(), "encoder_hidden_states");
    if (enc_hidden_state_port.has_value()) {
        m_is_whisper = true;
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
    fill_tensor_bytes(m_prefill_request->get_tensor(m_prefill_in_ports.at(m_input_ids_name)), 0u);
    fill_tensor<int64_t>(m_prefill_request->get_tensor(m_prefill_in_ports.at("attention_mask")), 0);
    if (!m_is_whisper) {
        fill_tensor<int64_t>(m_prefill_request->get_tensor(m_prefill_in_ports.at("position_ids")), 0);
    }

    fill_tensor<int64_t>(m_kvcache_request->get_tensor(m_kvcache_in_ports.at("attention_mask")), 0);
    m_npuw_llm_compiled_model->m_kvcache_desc.num_stored_tokens = 0u;
}

void ov::npuw::LLMInferRequest::infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                              ov::SoPtr<ov::ITensor> attention_mask,
                                              ov::SoPtr<ov::ITensor> position_ids,
                                              ov::SoPtr<ov::ITensor> enc_hidden_states) {
    LOG_DEBUG("Calling inference for prefill model...");
    LOG_BLOCK();

    prepare_for_new_conversation();

    if (m_is_whisper) {
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

        std::copy_n(reinterpret_cast<uint8_t*>(enc_hidden_states_data),
                    enc_hidden_states->get_byte_size(),
                    reinterpret_cast<uint8_t*>(encoder_hidden_states->data()));
    } else {  // LLM/VLM
        auto padded_input = m_prefill_request->get_tensor(m_prefill_in_ports.at(m_input_ids_name));
        // NB: padded_input can be either fp32(VLM) or i64(LLM)
        std::copy_n(reinterpret_cast<uint8_t*>(input_ids->data()),
                    input_ids->get_byte_size(),
                    reinterpret_cast<uint8_t*>(padded_input->data()) + padded_input->get_byte_size() -
                        input_ids->get_byte_size());

        auto padded_attention_mask = m_prefill_request->get_tensor(m_prefill_in_ports.at("attention_mask"));
        std::copy_n(
            attention_mask->data<int64_t>(),
            attention_mask->get_size(),
            padded_attention_mask->data<int64_t>() + padded_attention_mask->get_size() - attention_mask->get_size());

        auto padded_position_ids = m_prefill_request->get_tensor(m_prefill_in_ports.at("position_ids"));

        std::copy_n(position_ids->data<int64_t>(),
                    position_ids->get_size(),
                    padded_position_ids->data<int64_t>() + padded_position_ids->get_size() - position_ids->get_size());
    }

    m_prefill_request->infer();

    m_npuw_llm_compiled_model->m_kvcache_desc.num_stored_tokens +=
        static_cast<uint32_t>(input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM]);
    m_need_copy_kvcache = true;

    m_logits = m_is_whisper ? make_tensor_slice(m_prefill_request->get_tensor(m_prefill_out_ports.at("logits")),
                                                1u,
                                                0u,
                                                m_npuw_llm_compiled_model->m_kvcache_desc.num_stored_tokens)
                            : m_prefill_request->get_tensor(m_prefill_out_ports.at("logits"));

    LOG_DEBUG("Done");
}

void ov::npuw::LLMInferRequest::infer_generate(ov::SoPtr<ov::ITensor> input_ids,
                                               ov::SoPtr<ov::ITensor> attention_mask,
                                               ov::SoPtr<ov::ITensor> position_ids) {
    LOG_DEBUG("Calling inference for generate model...");
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

            if (m_is_whisper && output_name.find("decoder") == std::string::npos) {
                // Find self-attention KVcache tensors for Whisper, cross attn copy later
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
            fill_tensor<ov::float16>(kvcache_in_tensor, 0);

            const auto& kv_dim = (output_name.find("value") != std::string::npos && kvcache_desc.v_tensors_transposed)
                                     ? 3u
                                     : kvcache_desc.dim;

            auto prefill_out_slice =
                make_tensor_slice(prefill_out_tensor,
                                  kv_dim,
                                  m_is_whisper ? 0 : kvcache_desc.max_prompt_size - kvcache_desc.num_stored_tokens,
                                  m_is_whisper ? kvcache_desc.num_stored_tokens : kvcache_desc.max_prompt_size);

            auto kvcache_in_slice = make_tensor_slice(kvcache_in_tensor, kv_dim, 0u, kvcache_desc.num_stored_tokens);

            if (kv_dim == 3u) {
                copy_columns_by_row_chunks(prefill_out_slice, kvcache_in_slice);
            } else if (kv_dim == 2u) {
                copy_by_planes(prefill_out_slice, kvcache_in_slice);
            } else {
                prefill_out_slice->copy_to(kvcache_in_slice._ptr);
            }
        }

        if (m_is_whisper) {
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
        }

        LOG_DEBUG("Prepare attention mask pattern.");
        if (m_is_whisper) {
            // NB: Prepare attention mask to be in a format [0, 0, 0, 1, 1, 1, ..., 1, 0, 1]
            auto* attention_mask_data =
                m_kvcache_request->get_tensor(m_kvcache_in_ports.at("attention_mask"))->data<uint64_t>();
            auto attention_mask_size =
                m_kvcache_request->get_tensor(m_kvcache_in_ports.at("attention_mask"))->get_size();
            std::fill(attention_mask_data, attention_mask_data + kvcache_desc.num_stored_tokens, 0);
            std::fill(attention_mask_data + kvcache_desc.num_stored_tokens,
                      attention_mask_data + attention_mask_size - 2,
                      1);
            attention_mask_data[attention_mask_size - 2] = 0;
            attention_mask_data[attention_mask_size - 1] = 1;
        } else {
            auto* attention_mask_data =
                m_kvcache_request->get_tensor(m_kvcache_in_ports.at("attention_mask"))->data<int64_t>();
            attention_mask_data[kvcache_desc.total_size - 1] = 1;
        }

        m_need_copy_kvcache = false;
    }

    if (m_is_whisper) {
        auto kv_input_ids = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(m_input_ids_name));
        std::copy_n(input_ids->data<int64_t>(), input_ids->get_size(), kv_input_ids->data<int64_t>());

        auto kv_attn_mask = m_kvcache_request->get_tensor(m_kvcache_in_ports.at("attention_mask"));
        kv_attn_mask->data<int64_t>()[kvcache_desc.num_stored_tokens - 1] = 0;

        auto kv_cache_pos = m_kvcache_request->get_tensor(m_kvcache_in_ports.at("cache_position"));
        kv_cache_pos->data<int64_t>()[0] = kvcache_desc.num_stored_tokens;
    } else {
        // FIXME: these tensors should be shared between the parent & child models
        auto kv_input_ids = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(m_input_ids_name));
        // NB: input_ids can be either fp32(VLM) or i64(LLM)
        std::copy_n(reinterpret_cast<uint8_t*>(input_ids->data()),
                    input_ids->get_byte_size(),
                    reinterpret_cast<uint8_t*>(kv_input_ids->data()));

        auto kv_attn_mask = m_kvcache_request->get_tensor(m_kvcache_in_ports.at("attention_mask"));
        std::copy_n(attention_mask->data<int64_t>(), attention_mask->get_size() - 1, kv_attn_mask->data<int64_t>());

        auto kv_pos_ids = m_kvcache_request->get_tensor(m_kvcache_in_ports.at("position_ids"));
        std::copy_n(position_ids->data<int64_t>(), position_ids->get_size(), kv_pos_ids->data<int64_t>());
    }

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
        auto kvcache_in_slice = make_tensor_slice(kvcache_in_tensor,
                                                  kv_dim,
                                                  kvcache_desc.num_stored_tokens - 1,
                                                  kvcache_desc.num_stored_tokens);
        auto kvcache_out_tensor = m_kvcache_request->get_tensor(m_kvcache_out_ports.at(output_name));
        if (kv_dim == 3u) {
            ov::npuw::util::XARCH::copy_row_as_column(kvcache_out_tensor, kvcache_in_slice);
        } else if (kv_dim == 2u) {
            copy_by_planes(kvcache_out_tensor, kvcache_in_slice);
        } else {
            kvcache_out_tensor->copy_to(kvcache_in_slice._ptr);
        }
    }
    LOG_DEBUG("Done");
}

void ov::npuw::LLMInferRequest::infer() {
    const auto& inputs = get_inputs();

    auto input_ids = get_tensor(find_port_by_name(inputs, m_input_ids_name).value());
    // NB: For VLM, the "inputs_embeds" contains float values (embeddings)
    OPENVINO_ASSERT(ov::element::f32 == input_ids->get_element_type() ||
                    ov::element::i64 == input_ids->get_element_type());

    ov::SoPtr<ov::ITensor> attention_mask;         // LLM/VLM input
    ov::SoPtr<ov::ITensor> position_ids;           // LLM/VLM input
    ov::SoPtr<ov::ITensor> encoder_hidden_states;  // Whisper input

    if (m_is_whisper) {
        encoder_hidden_states = get_tensor(find_port_by_name(inputs, "encoder_hidden_states").value());
        OPENVINO_ASSERT(ov::element::f32 == encoder_hidden_states->get_element_type());
    } else {
        attention_mask = get_tensor(find_port_by_name(inputs, "attention_mask").value());
        position_ids = get_tensor(find_port_by_name(inputs, "position_ids").value());

        OPENVINO_ASSERT(ov::element::i64 == attention_mask->get_element_type());
        OPENVINO_ASSERT(ov::element::i64 == position_ids->get_element_type());
    }

    // NB: Check the sequence length provided for input_ids
    // in order to distinguish prefill / generate stages
    if (input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM] != 1) {
        infer_prefill(input_ids, attention_mask, position_ids, encoder_hidden_states);
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
