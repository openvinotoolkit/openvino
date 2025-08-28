// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_infer_request.hpp"

#include <cstddef>
#include <iostream>
#include <ostream>
#include <regex>

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

    const auto& src_shape = src->get_shape();

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

void copy_tensor_by_dim(ov::SoPtr<ov::ITensor> src_tensor, ov::SoPtr<ov::ITensor> dst_tensor, uint32_t kv_dim) {
    if (kv_dim == 3u) {
        // Asserting that we work with last dimenston here:
        const auto& src_shape = src_tensor->get_shape();
        OPENVINO_ASSERT(src_shape.size() == 4);
        // If last dimenstion of src_tensor is equal to 1, then we can squeeze
        // src_shape from [1, heads, d_v, seq_len=1] to [heads, d_v].
        // We can then treat src_tensor as a continuous tensor of row value vectors
        // for multiple heads, while dst_tensor will still have [1, heads, d_v, seq_len!=1],
        // shape, awaiting updates at column dimension, as value vectors are columns now.
        if (src_shape[kv_dim] == 1 && src_tensor->is_continuous()) {
            ov::npuw::util::XARCH::copy_row_as_column(src_tensor, dst_tensor);
        } else {
            copy_columns_by_row_chunks(src_tensor, dst_tensor);
        }
    } else if (kv_dim == 2u) {
        copy_by_planes(src_tensor, dst_tensor);
    } else {
        src_tensor->copy_to(dst_tensor._ptr);
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

void copy_columns_by_row_chunks_2d(ov::SoPtr<ov::ITensor> src, ov::SoPtr<ov::ITensor>& dst) {
    const auto& src_shape = src->get_shape();

    OPENVINO_ASSERT(src_shape.size() == 2u);
    OPENVINO_ASSERT(src_shape == dst->get_shape());
    OPENVINO_ASSERT(src->get_byte_size() == dst->get_byte_size());

    const auto& src_strides = src->get_strides();
    const auto& dst_strides = dst->get_strides();
    const auto elem_size = src->get_byte_size() / src->get_size();

    const auto H = src_shape[0];
    const auto W = src_shape[1];

    const auto IS_H = src_strides[0];
    const auto OS_H = dst_strides[0];

    const size_t chunk_byte_size = W * elem_size;

    const auto* src_p = static_cast<uint8_t*>(src->data());
    auto* dst_p = static_cast<uint8_t*>(dst->data());

    for (size_t i = 0; i < H; ++i) {
        const size_t src_offset = i * IS_H;
        const size_t dst_offset = i * OS_H;
        std::copy_n(src_p + src_offset, chunk_byte_size, dst_p + dst_offset);
    }
}

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

void check_tensor_shape_compatibility(const ov::Shape& state_tensor_shape,
                                      const ov::Shape& infer_tensor_shape,
                                      size_t full_rank_dim,
                                      size_t low_rank_dim,
                                      uint32_t max_low_rank_dim_size) {
    if (state_tensor_shape[full_rank_dim] != infer_tensor_shape[full_rank_dim]) {
        OPENVINO_THROW("LoRA adapter tensor shape: ",
                       state_tensor_shape,
                       " is not compatible with inference tensor shape: ",
                       infer_tensor_shape,
                       ". Please check if adapter is compatible with the base model.");
    }

    uint32_t state_tensor_low_rank_size = static_cast<uint32_t>(state_tensor_shape[low_rank_dim]);
    if (state_tensor_low_rank_size > max_low_rank_dim_size) {
        OPENVINO_THROW("LoRA tensor low-rank size: ",
                       state_tensor_low_rank_size,
                       " is larger than the maximum LoRA low-rank size ",
                       max_low_rank_dim_size,
                       ". Please adjust NPUW_LLM_MAX_LORA_RANK configuration.");
    }
}

std::pair<uint32_t, uint32_t> get_lora_dims_by_name(const std::string& state_name) {
    uint32_t low_rank_dim, full_rank_dim;
    if (ov::npuw::util::matchLoRAMatMulAString(state_name)) {
        // Shape of A is [r, d]
        low_rank_dim = 0;
        full_rank_dim = 1;
    } else if (ov::npuw::util::matchLoRAMatMulBString(state_name)) {
        // Shape of B is [d, r]
        low_rank_dim = 1;
        full_rank_dim = 0;
    } else if (ov::npuw::util::matchLoRAMatMulAlphaString(state_name)) {
        // Shape of alpha is [1, r]
        low_rank_dim = 1;
        full_rank_dim = 0;
    } else {
        OPENVINO_THROW("Unknown LoRA state name: " + state_name);
    }

    return std::make_pair(low_rank_dim, full_rank_dim);
}

constexpr uint32_t INPUT_IDS_SEQ_LEN_DIM = 1;

constexpr std::size_t kStartOutputKVCacheLayers = 1;

}  // anonymous namespace

void ov::npuw::LLMInferRequest::init_lora_states() {
    // For KV chunking prefill models, only the first request is used for LoRA state initialization.
    // All chunked prefill requests share the same IR topology and input ports for LoRA parameters,
    // so initializing variable states from the first request is sufficient and avoids redundant setup.
    // This ensures LoRA weights are correctly managed across all chunked prefill requests without duplication.
    auto request = m_prefill_requests.front();
    for (const auto& input_port : request->get_compiled_model()->inputs()) {
        auto input_name = input_port.get_any_name();
        if (ov::npuw::util::matchLoRAMatMulAString(input_name) || ov::npuw::util::matchLoRAMatMulBString(input_name) ||
            ov::npuw::util::matchLoRAMatMulAlphaString(input_name)) {
            auto input_tensor = request->get_tensor(input_port);
            m_variableStates.push_back(std::make_shared<VariableState>(input_name, input_tensor));
        }
    }
}

ov::npuw::LLMInferRequest::LLMInferRequest(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_npuw_llm_compiled_model(compiled_model) {
    std::cout << "LLMInferRequest::LLMInferRequest" << std::endl;
    for (const auto& input_port : m_npuw_llm_compiled_model->inputs()) {
        init_tensor(input_port);
    }
    for (const auto& output_port : m_npuw_llm_compiled_model->outputs()) {
        init_tensor(output_port);
    }

    // --- Unified prefill context setup ---
    auto setup_prefill_context = [&](auto& compiled_model_ref) {
        auto& prefill_models = compiled_model_ref->m_prefill_compiled;
        auto input_ids_port = find_port_by_name(prefill_models.back()->inputs(), layer_names::input_ids);
        m_input_ids_name = input_ids_port.has_value() ? layer_names::input_ids : layer_names::inputs_embeds;
        for (auto model : prefill_models) {
            auto request = model->create_infer_request();
            m_prefill_requests.push_back(request);
        }
        for (auto request : m_prefill_requests) {
            std::unordered_map<std::string, ov::Output<const ov::Node>> in_ports_map;
            for (const auto& input_port : request->get_compiled_model()->inputs()) {
                in_ports_map.emplace(input_port.get_any_name(), input_port);
            }
            m_prefill_in_ports.push_back(in_ports_map);

            std::unordered_map<std::string, ov::Output<const ov::Node>> out_ports_map;
            for (const auto& output_port : request->get_compiled_model()->outputs()) {
                out_ports_map.emplace(output_port.get_any_name(), output_port);
            }
            m_prefill_out_ports.push_back(out_ports_map);
        }
    };
    m_kvcache_request = compiled_model->m_kvcache_compiled->create_infer_request();
    setup_prefill_context(compiled_model);

    for (const auto& input_port : m_kvcache_request->get_compiled_model()->inputs()) {
        m_kvcache_in_ports.emplace(input_port.get_any_name(), input_port);
    }
    for (const auto& output_port : m_kvcache_request->get_compiled_model()->outputs()) {
        m_kvcache_out_ports.emplace(output_port.get_any_name(), output_port);
    }

    init_lora_states();

    if (m_npuw_llm_compiled_model->m_use_chunk_prefill) {
        clear_chunk_prefill_kv_cache();
    }

    if (compiled_model->m_lm_head_compiled) {
        m_lm_head_request = compiled_model->m_lm_head_compiled->create_infer_request();
        OPENVINO_ASSERT(m_lm_head_request);
        const ov::Output<const ov::Node> lm_head_embed_port = m_lm_head_request->get_inputs()[0];
        m_lm_head_logits_port = m_lm_head_request->get_outputs()[0];

        if (compiled_model->m_prefill_compiled.size() == 1) {
            // For non-chunked KV (single prefill model), the LM head input tensor can be directly shared
            // between the prefill and KV cache infer requests.
            // This guarantees that output embeddings are consistently linked for both prefill and generation phases.
            // But in chunked KV mode, multiple prefill requests may produce output embeddings from different chunks.
            // In this case, sharing the LM head tensor must be handled after all chunked prefill requests are
            // processed, to avoid ambiguity and ensure correct tensor linkage.
            m_prefill_requests.front()->set_tensor(m_prefill_out_ports.front().at(layer_names::output_embeds),
                                                   m_lm_head_request->get_tensor(lm_head_embed_port));

            m_kvcache_request->set_tensor(m_kvcache_out_ports.at(layer_names::output_embeds),
                                          m_lm_head_request->get_tensor(lm_head_embed_port));
        }
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

    ov::Any kvcache_weight_bank_alloc =
        compiled_model->m_kvcache_compiled->get_property(ov::intel_npu::npuw::weights_bank_alloc.name());
    if (kvcache_weight_bank_alloc.as<std::string>() == "CPU") {
        enable_cpu_wa = true;
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
            fill_tensor<ov::float16>(kvcache_in_tensor, 0);
        }
    }

    std::cout << "LLMInferRequest::LLMInferRequest is created" << std::endl;

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

void ov::npuw::LLMInferRequest::apply_lora() {
    uint32_t max_low_rank_dim_size = m_npuw_llm_compiled_model->m_max_lora_rank;

    bool pre_alloc_on_npu = true;
    const auto& prefill_compiled = m_npuw_llm_compiled_model->m_prefill_compiled.front();
    for (std::size_t idx = 0; idx < prefill_compiled->m_compiled_submodels.size(); ++idx) {
        if (prefill_compiled->submodel_device(idx) != "NPU") {
            pre_alloc_on_npu = false;
            break;
        }
    }
    std::string device = pre_alloc_on_npu ? "NPU" : "CPU";

    for (auto state : m_variableStates) {
        auto state_name = state->get_name();
        auto state_tensor = state->get_state();

        auto variableState = dynamic_cast<VariableState*>(state.operator->());
        if (!variableState) {
            OPENVINO_THROW("Failed to cast ov::IVariableState to VariableState.");
        }

        bool stateUpdated = variableState->is_state_updated();
        if (!stateUpdated) {
            continue;
        }

        if (state_tensor->get_size() == 0) {
            // Generate without LoRA:
            // the size of applied LoRA tensor from GenAI is 0

            // Disable adapter by setting alpha to 0
            if (ov::npuw::util::matchLoRAMatMulAlphaString(state_name)) {
                for (size_t i = 0; i < m_prefill_requests.size(); ++i) {
                    auto& request = m_prefill_requests[i];
                    auto& in_ports = m_prefill_in_ports[i];
                    auto prefill_lora_in_tensor = request->get_tensor(in_ports.at(state_name));
                    fill_tensor<float>(prefill_lora_in_tensor, 0.0f);
                }

                auto kvcach_lora_in_tensor = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(state_name));
                fill_tensor<float>(kvcach_lora_in_tensor, 0.0f);
            }
        } else {
            // Generate with LoRA
            auto infer_tensor_shape =
                m_prefill_requests.front()->get_tensor(m_prefill_in_ports.front().at(state_name))->get_shape();
            auto state_tensor_shape = state_tensor->get_shape();
            auto lora_dims = get_lora_dims_by_name(state_name);
            auto low_rank_dim = std::get<0>(lora_dims);
            auto full_rank_dim = std::get<1>(lora_dims);

            check_tensor_shape_compatibility(state_tensor_shape,
                                             infer_tensor_shape,
                                             full_rank_dim,
                                             low_rank_dim,
                                             max_low_rank_dim_size);

            uint32_t state_tensor_rank = static_cast<uint32_t>(state_tensor_shape[low_rank_dim]);
            uint32_t target_lora_rank = static_cast<uint32_t>(infer_tensor_shape[low_rank_dim]);

            auto prefill_lora_in_tensor =
                m_prefill_requests.front()->get_tensor(m_prefill_in_ports.front().at(state_name));
            auto new_infer_tensor = ov::npuw::util::allocMem(prefill_lora_in_tensor->get_element_type(),
                                                             prefill_lora_in_tensor->get_shape(),
                                                             device,
                                                             m_npuw_llm_compiled_model->get_plugin());
            bool has_padding = state_tensor_rank != target_lora_rank;
            if (has_padding) {
                // Clear padding tensor in infer request
                fill_tensor<float>(new_infer_tensor, 0.0f);
            }

            // Fill LoRA into infer request
            auto fill_lora_in_tensor = [low_rank_dim, state_tensor_rank](ov::SoPtr<ov::ITensor> state_tensor,
                                                                         ov::SoPtr<ov::ITensor> infer_tensor,
                                                                         bool has_padding) {
                if (!has_padding) {
                    state_tensor->copy_to(infer_tensor._ptr);
                    return;
                }

                auto new_tensor_slice = make_tensor_slice(infer_tensor, low_rank_dim, 0u, state_tensor_rank);
                if (low_rank_dim == 1) {
                    copy_columns_by_row_chunks_2d(state_tensor, new_tensor_slice);
                } else {
                    state_tensor->copy_to(new_tensor_slice._ptr);
                }
            };
            fill_lora_in_tensor(state_tensor, new_infer_tensor, has_padding);

            // Set new tensor for inference
            for (size_t i = 0; i < m_prefill_requests.size(); ++i) {
                auto& request = m_prefill_requests[i];
                auto& in_ports = m_prefill_in_ports[i];
                request->set_tensor(in_ports.at(state_name), new_infer_tensor);
            }
            m_kvcache_request->set_tensor(m_kvcache_in_ports.at(state_name), new_infer_tensor);
        }
        variableState->clear_state_updated();
    }
}

void ov::npuw::LLMInferRequest::prepare_for_new_conversation() {
    auto init_prefill = [&](auto& request, auto& in_ports) {
        fill_tensor_bytes(request->get_tensor(in_ports.at(m_input_ids_name)), 0u);
        fill_tensor<int64_t>(request->get_tensor(in_ports.at(layer_names::attention_mask)), 0);
        fill_tensor<int64_t>(request->get_tensor(in_ports.at(layer_names::position_ids)), 0);
    };

    for (size_t i = 0; i < m_prefill_requests.size(); ++i) {
        auto& request = m_prefill_requests[i];
        auto& in_ports = m_prefill_in_ports[i];
        init_prefill(request, in_ports);
    }

    m_npuw_llm_compiled_model->m_kvcache_desc.num_stored_tokens = 0u;

    std::cout << "apply_lora." << std::endl;

    apply_lora();
}

void ov::npuw::LLMInferRequest::copy_kvcache() {
    LOG_DEBUG("Copying kv-cache from prefill to generate model.");
    LOG_BLOCK();
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    const auto& kvcache_compiled = m_kvcache_request->get_compiled_model();
    // FIXME: Find only matching by names outputs and copy them, having previously checked that such inputs exist
    for (std::size_t i = kStartOutputKVCacheLayers; i < kvcache_compiled->outputs().size(); ++i) {
        const auto& output_name = kvcache_compiled->outputs()[i].get_any_name();
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

            // Copy part 1 KV results - past KV

            // For Q chunked without chunked KV (single prefill model), the past KV results are always stored in
            // m_prefill_requests[0]. In the chunked KV scenario, only the last chunked prefill request contains the
            // complete past KV results for all processed tokens, while earlier requests hold partial results.
            // Therefore, when copying past KV, use the last chunked prefill request to ensure all relevant tokens are
            // included.
            size_t src_prefill_model_index = m_prefill_requests.size() == 1 ? 0 : m_last_infer_chunk_idx;
            auto tokens_in_past_chunks = kvcache_desc.num_stored_tokens - m_tokens_in_present_chunk;
            if (tokens_in_past_chunks > 0) {
                auto prefill_past_kv = m_prefill_requests[src_prefill_model_index]->get_tensor(
                    m_prefill_in_ports[src_prefill_model_index].at(input_name));
                auto prefill_past_kv_chunks =
                    make_tensor_slice(prefill_past_kv, kv_dim, 0u, static_cast<uint32_t>(tokens_in_past_chunks));

                auto kvcache_past_kv_chunks =
                    make_tensor_slice(kvcache_in_tensor, kv_dim, 0u, static_cast<uint32_t>(tokens_in_past_chunks));

                copy_tensor_by_dim(prefill_past_kv_chunks, kvcache_past_kv_chunks, kv_dim);
            }

            // Copy part 2 KV results - present KV

            // For Q chunked without chunked KV (single prefill model), the present KV results
            // are stored in m_prefill_requests[0].
            // In chunked KV mode, only the last chunked prefill request produces the final present KV results for the
            // most recent chunk of tokens. Earlier requests do not contain valid present KV for the current decoding
            // step. Always use the last chunked prefill request to obtain the correct present KV results for downstream
            // processing.
            src_prefill_model_index = m_prefill_requests.size() == 1 ? 0 : m_last_infer_chunk_idx;
            auto prefill_out_tensor = m_prefill_requests[src_prefill_model_index]->get_tensor(
                m_prefill_out_ports[src_prefill_model_index].at(output_name));
            auto prefill_present_kv_chunk =
                make_tensor_slice(prefill_out_tensor,
                                  kv_dim,
                                  static_cast<uint32_t>(prefill_chunk_size - m_tokens_in_present_chunk),
                                  static_cast<uint32_t>(prefill_chunk_size));

            auto kvcache_last_kv_chunk = make_tensor_slice(kvcache_in_tensor,
                                                           kv_dim,
                                                           static_cast<uint32_t>(tokens_in_past_chunks),
                                                           kvcache_desc.num_stored_tokens);

            copy_tensor_by_dim(prefill_present_kv_chunk, kvcache_last_kv_chunk, kv_dim);
        } else {
            // Without chunk prefill, we only have single prefill model
            auto prefill_out_tensor =
                m_prefill_requests.front()->get_tensor(m_prefill_out_ports.front().at(output_name));
            auto prefill_out_slice = make_tensor_slice(prefill_out_tensor,
                                                       kv_dim,
                                                       kvcache_desc.max_prompt_size - kvcache_desc.num_stored_tokens,
                                                       kvcache_desc.max_prompt_size);

            auto kvcache_in_slice = make_tensor_slice(kvcache_in_tensor, kv_dim, 0u, kvcache_desc.num_stored_tokens);

            copy_tensor_by_dim(prefill_out_slice, kvcache_in_slice, kv_dim);
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
        auto dst_slice = make_tensor_slice(dst_tensor,
                                           kv_dim,
                                           kvcache_desc.num_stored_tokens - num_tokens,
                                           kvcache_desc.num_stored_tokens);
        auto src_tensor = request->get_tensor(out_ports.at(output_name));
        copy_tensor_by_dim(src_tensor, dst_slice, kv_dim);
    }
    LOG_DEBUG("Done.");
}

void ov::npuw::LLMInferRequest::update_kvcache_between_chunks(uint32_t num_tokens, size_t chunk_idx) {
    LOG_DEBUG("Updating KVCache between chunk models: copying present and past KV tensors to next chunk's input.");
    LOG_BLOCK();

    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    auto& compiled = m_prefill_requests[chunk_idx]->get_compiled_model();

    // FIXME: Find only matching by names outputs and copy them, having previously checked that such inputs exist
    for (std::size_t i = kStartOutputKVCacheLayers; i < compiled->outputs().size(); ++i) {
        const auto& output_name = compiled->outputs()[i].get_any_name();
        const auto& input_name = std::regex_replace(output_name, std::regex("present"), layer_names::past_key_values);
        auto& cur_in_ports = m_prefill_in_ports[chunk_idx];
        auto& next_in_ports = m_prefill_in_ports[chunk_idx + 1];
        auto& cur_out_ports = m_prefill_out_ports[chunk_idx];

        if (cur_in_ports.find(input_name) == cur_in_ports.end() ||
            next_in_ports.find(input_name) == next_in_ports.end()) {
            LOG_DEBUG("Input name " << input_name << " not found in chunk ports. Skipping.");
            continue;
        }

        // Step 1: Copy present KV (newly computed tokens) from current chunk's output to next chunk's past KV input
        auto dst_tensor = m_prefill_requests[chunk_idx + 1]->get_tensor(next_in_ports.at(input_name));
        const auto& kv_dim = (output_name.find("value") != std::string::npos && kvcache_desc.v_tensors_transposed)
                                 ? 3u
                                 : kvcache_desc.dim;
        auto dst_slice = make_tensor_slice(dst_tensor,
                                           kv_dim,
                                           kvcache_desc.num_stored_tokens - num_tokens,
                                           kvcache_desc.num_stored_tokens);
        auto src_tensor = m_prefill_requests[chunk_idx]->get_tensor(cur_out_ports.at(output_name));

        if (src_tensor->get_shape()[kv_dim] != dst_slice->get_shape()[kv_dim]) {
            // When prefix caching is enabled, there might be padding data on the left side after the first chunking
            // inference
            auto src_slice = make_tensor_slice(src_tensor,
                                               kv_dim,
                                               static_cast<uint32_t>(src_tensor->get_shape()[kv_dim]) - num_tokens,
                                               static_cast<uint32_t>(src_tensor->get_shape()[kv_dim]));
            copy_tensor_by_dim(src_slice, dst_slice, kv_dim);
        } else {
            copy_tensor_by_dim(src_tensor, dst_slice, kv_dim);
        }

        // For the first chunk, there is no valid past KV to copy
        if (chunk_idx == 0) {
            LOG_DEBUG("First chunk: skipping past KV copy.");
            continue;
        }

        // Step 2: Copy past KV (previously stored tokens) from current chunk's input to next chunk's past KV input
        auto past_dst_slice = make_tensor_slice(dst_tensor, kv_dim, 0u, kvcache_desc.num_stored_tokens - num_tokens);
        auto past_src_tensor = m_prefill_requests[chunk_idx]->get_tensor(cur_in_ports.at(input_name));
        if (past_src_tensor->get_shape()[kv_dim] != past_dst_slice->get_shape()[kv_dim]) {
            auto past_src_slice =
                make_tensor_slice(past_src_tensor, kv_dim, 0u, kvcache_desc.num_stored_tokens - num_tokens);
            copy_tensor_by_dim(past_src_slice, past_dst_slice, kv_dim);
        } else {
            copy_tensor_by_dim(past_src_tensor, past_dst_slice, kv_dim);
        }
    }

    LOG_DEBUG("KVCache update between chunks complete.");
}

void ov::npuw::LLMInferRequest::clear_chunk_prefill_kv_cache() {
    auto fill_chunk_kvcache = [&](auto& request, auto& in_ports) {
        auto model = request->get_compiled_model();
        for (std::size_t i = kStartOutputKVCacheLayers; i < model->outputs().size(); ++i) {
            const auto& output_name = model->outputs()[i].get_any_name();
            const auto& input_name = std::regex_replace(output_name, std::regex("present"), "past_key_values");
            if (in_ports.find(input_name) == in_ports.end()) {
                LOG_DEBUG("Input name " << input_name << " doesn't contain kv cache. Skipping.");
                continue;
            }
            auto chunk_prefill_kvcache_in_tensor = request->get_tensor(in_ports.at(input_name));
            fill_tensor<ov::float16>(chunk_prefill_kvcache_in_tensor, 0);
        }
    };

    for (size_t i = 0; i < m_prefill_requests.size(); ++i) {
        fill_chunk_kvcache(m_prefill_requests[i], m_prefill_in_ports[i]);
    }
}

void ov::npuw::LLMInferRequest::infer_chunked_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                                      ov::SoPtr<ov::ITensor> attention_mask,
                                                      ov::SoPtr<ov::ITensor> position_ids) {
    LOG_DEBUG("Calling chunked inference for prefill model.");
    LOG_BLOCK();
    std::cout << "Calling chunked inference for prefill model." << std::endl;

    const auto input_prompt_len = input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM];

    // For LLM, model accepts 2d inputs_embeds[BATCH, SEQ_LEN]
    // For VLM, model accepts 3d inputs_ids[BATCH, SEQ_LEN, EMB_SIZE]
    bool is_input_embeds = input_ids->get_shape().size() == 2 ? false : true;

    const auto input_ids_elem_size = input_ids->get_element_type().size();
    const int64_t chunk_prompt_len = m_npuw_llm_compiled_model->m_prefill_chunk_size;

    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;

    int64_t remaining_prompts = input_prompt_len;
    size_t chunk_id = 0;
    while (remaining_prompts > 0) {
        std::cout << "infer chunk " << chunk_id << std::endl;
        // NB: input_ids can be either fp32(VLM) or i64(LLM)
        // The last chunk may not be completely filled if the actual length of the prompts is not evenly divisible by
        // the chunk size
        auto current_prompts_len = std::min(remaining_prompts, chunk_prompt_len);

        std::cout << "Populate the attention mask for the present chunk" << std::endl;
        // Populate the attention mask for the present chunk
        // For the already processed tokens, they will be added into the attention mask after inference call
        size_t prefill_model_index = m_prefill_requests.size() == 1 ? 0 : chunk_id;
        auto attn_mask_in_tensor = m_prefill_requests[prefill_model_index]->get_tensor(
            m_prefill_in_ports[prefill_model_index].at(layer_names::attention_mask));

        size_t last_chunk_offset = attn_mask_in_tensor->get_size() - chunk_prompt_len;
        if (current_prompts_len < chunk_prompt_len) {
            // We will populate current_prompts_len on the right side of attention mask for the processing tokens
            // If the current prompt length is smaller than the chunk prompt length,
            // clear the last chunk of the attention mask to ensure non-relevant tokens are masked
            fill_tensor<int64_t>(attn_mask_in_tensor, 0, last_chunk_offset);
        }
        std::copy_n(attention_mask->data<int64_t>() + kvcache_desc.num_stored_tokens,
                    current_prompts_len,
                    attn_mask_in_tensor->data<int64_t>() + attn_mask_in_tensor->get_size() - current_prompts_len);

        auto current_prefill_bytes = current_prompts_len * input_ids_elem_size;
        auto prefilled_bytes = kvcache_desc.num_stored_tokens * input_ids_elem_size;
        if (is_input_embeds) {
            current_prefill_bytes *= input_ids->get_shape().back();
            prefilled_bytes *= input_ids->get_shape().back();
        }

        std::cout << "Populate input ids" << std::endl;
        auto input_ids_in_tensor = m_prefill_requests[prefill_model_index]->get_tensor(
            m_prefill_in_ports[prefill_model_index].at(m_input_ids_name));
        std::copy_n(reinterpret_cast<uint8_t*>(input_ids->data()) + prefilled_bytes,
                    current_prefill_bytes,
                    reinterpret_cast<uint8_t*>(input_ids_in_tensor->data()) + input_ids_in_tensor->get_byte_size() -
                        current_prefill_bytes);

        // NB: Regular LLM uses 2D position_ids [BATCH, SEQ_LEN], Qwen2.5 VL/Omni uses 3D position_ids [3, BATCH,
        // SEQ_LEN]
        // Copy postion ids with considering the 3D position_ids
        std::cout << "pad_position_ids" << std::endl;
        auto last_dim = position_ids->get_shape().size() - 1;
        auto actual_position_ids_slice =
            make_tensor_slice(position_ids,
                              static_cast<uint32_t>(last_dim),
                              kvcache_desc.num_stored_tokens,
                              kvcache_desc.num_stored_tokens + static_cast<uint32_t>(current_prompts_len));
        auto pos_ids_in_tensor = m_prefill_requests[prefill_model_index]->get_tensor(
            m_prefill_in_ports[prefill_model_index].at(layer_names::position_ids));
        pad_position_ids(pos_ids_in_tensor, actual_position_ids_slice);

        std::cout << "start infer" << std::endl;
        m_prefill_requests[prefill_model_index]->infer();
        // Record the last infer model index
        m_last_infer_chunk_idx = prefill_model_index;

        remaining_prompts -= current_prompts_len;
        kvcache_desc.num_stored_tokens += static_cast<uint32_t>(current_prompts_len);

        chunk_id++;

        // Do not copy last computed chunk and preserve it in present k/v layer
        if (remaining_prompts <= 0) {
            LOG_DEBUG("All prompts have been prefilled in chunks");
            std::cout << "All prompts have been prefilled in chunks, m_last_infer_chunk_idx: " << m_last_infer_chunk_idx
                      << std::endl;
            m_tokens_in_present_chunk = current_prompts_len;
            break;
        }

        if (m_prefill_requests.size() == 1) {
            // Copy calculated key/values chunk from present k/v layer to past k/v layer for storage
            update_kvcache_for(m_prefill_requests[0],
                               m_prefill_in_ports[0],
                               m_prefill_out_ports[0],
                               static_cast<uint32_t>(current_prompts_len));

            // Update attention mask for the next iteration
            std::copy_n(attn_mask_in_tensor->data<int64_t>() + attn_mask_in_tensor->get_size() - current_prompts_len,
                        current_prompts_len,
                        attn_mask_in_tensor->data<int64_t>() + kvcache_desc.num_stored_tokens - current_prompts_len);
        } else {
            std::cout << "Infer done: update for next chunk" << std::endl;

            // Copy calculated key/values chunk from present k/v layer of current chunk model to past k/v layer of next
            // chunk model
            update_kvcache_between_chunks(static_cast<uint32_t>(current_prompts_len), chunk_id - 1);

            // Update attention mask for the next iteration
            auto attn_mask_in_tensor_of_next_chunk =
                m_prefill_requests[chunk_id]->get_tensor(m_prefill_in_ports[chunk_id].at(layer_names::attention_mask));

            std::copy_n(attention_mask->data<int64_t>(),
                        kvcache_desc.num_stored_tokens,
                        attn_mask_in_tensor_of_next_chunk->data<int64_t>());
        }
    }

    LOG_DEBUG("Done.");
}

void ov::npuw::LLMInferRequest::infer_whole_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                                    ov::SoPtr<ov::ITensor> attention_mask,
                                                    ov::SoPtr<ov::ITensor> position_ids) {
    LOG_DEBUG("Calling inference for prefill model in a single launch.");
    LOG_BLOCK();

    // NB: padded_input can be either fp32(VLM) or i64(LLM)
    auto padded_input = m_prefill_requests.front()->get_tensor(m_prefill_in_ports.front().at(m_input_ids_name));
    std::copy_n(
        reinterpret_cast<uint8_t*>(input_ids->data()),
        input_ids->get_byte_size(),
        reinterpret_cast<uint8_t*>(padded_input->data()) + padded_input->get_byte_size() - input_ids->get_byte_size());

    auto padded_attention_mask =
        m_prefill_requests.front()->get_tensor(m_prefill_in_ports.front().at(layer_names::attention_mask));
    std::copy_n(
        attention_mask->data<int64_t>(),
        attention_mask->get_size(),
        padded_attention_mask->data<int64_t>() + padded_attention_mask->get_size() - attention_mask->get_size());

    auto padded_position_ids =
        m_prefill_requests.front()->get_tensor(m_prefill_in_ports.front().at(layer_names::position_ids));
    pad_position_ids(padded_position_ids, position_ids);

    m_prefill_requests.front()->infer();
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    kvcache_desc.num_stored_tokens += static_cast<uint32_t>(input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM]);

    LOG_DEBUG("Done");
}

void ov::npuw::LLMInferRequest::infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                              ov::SoPtr<ov::ITensor> attention_mask,
                                              ov::SoPtr<ov::ITensor> position_ids) {
    LOG_DEBUG("Calling inference for prefill model...");
    LOG_BLOCK();
    std::cout << "Calling inference for prefill model..." << std::endl;

    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    if (input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM] > kvcache_desc.max_prompt_size) {
        OPENVINO_THROW("Input prompt is longer than configured \"NPUW_LLM_MAX_PROMPT_LEN\": ",
                       kvcache_desc.max_prompt_size,
                       ".\nPlease either setup bigger "
                       "\"NPUW_LLM_MAX_PROMPT_LEN\" or shorten the prompt.");
    }

    std::cout << "prepare_for_new_conversation..." << std::endl;
    prepare_for_new_conversation();

    const bool use_chunk_prefill = m_npuw_llm_compiled_model->m_use_chunk_prefill;
    if (use_chunk_prefill) {
        infer_chunked_prefill(input_ids, attention_mask, position_ids);
    } else {
        infer_whole_prefill(input_ids, attention_mask, position_ids);
    }

    std::cout << "Prefill infer done..." << std::endl;

    if (m_lm_head_request) {
        LOG_DEBUG("Calling inference for LM head model.");
        std::cout << "Calling inference for LM head model..." << std::endl;
        if (m_prefill_requests.size() > 1) {
            // For KV chunk, after chunked prefill, update the LM head request input tensor using the output embeddings
            // from the last chunk. This ensures the LM head receives the latest embeddings, and the KVCache infer
            // request output tensor is also synchronized.
            const auto& shared_output_embeds = m_prefill_requests[m_last_infer_chunk_idx]->get_tensor(
                m_prefill_out_ports[m_last_infer_chunk_idx].at(layer_names::output_embeds));
            const ov::Output<const ov::Node> lm_head_embed_port = m_lm_head_request->get_inputs()[0];
            m_lm_head_request->set_tensor(lm_head_embed_port, shared_output_embeds);
            m_kvcache_request->set_tensor(m_kvcache_out_ports.at(layer_names::output_embeds), shared_output_embeds);
        }
        m_lm_head_request->infer();
        m_logits = m_lm_head_request->get_tensor(m_lm_head_logits_port);
    } else {
        std::cout << "Not LLM head..." << std::endl;
        size_t prefill_model_index = m_prefill_requests.size() == 0 ? 0 : m_last_infer_chunk_idx;
        m_logits = m_prefill_requests[prefill_model_index]->get_tensor(
            m_prefill_out_ports[prefill_model_index].at(layer_names::logits));
    }

    m_generate_initialized = false;

    LOG_DEBUG("Done");
    std::cout << "Done" << std::endl;
}

void ov::npuw::LLMInferRequest::infer_generate(ov::SoPtr<ov::ITensor> input_ids,
                                               ov::SoPtr<ov::ITensor> attention_mask,
                                               ov::SoPtr<ov::ITensor> position_ids) {
    LOG_DEBUG("Calling inference for generate model...");
    LOG_BLOCK();

    std::cout << "Calling inference for generate model" << std::endl;

    if (!m_generate_initialized) {
        LOG_DEBUG("Copy kv-cache from prefill to generate model.");
        copy_kvcache();

        LOG_DEBUG("Prepare attention mask pattern.");
        auto kv_attn_mask = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(layer_names::attention_mask));
        fill_tensor<int64_t>(kv_attn_mask, 0);
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

    auto input_ids = get_tensor(find_port_by_name(inputs, m_input_ids_name).value());
    auto attention_mask = get_tensor(find_port_by_name(inputs, layer_names::attention_mask).value());
    // FIXME: position_ids might be optional for some models!
    auto position_ids = get_tensor(find_port_by_name(inputs, layer_names::position_ids).value());

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

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::LLMInferRequest::query_state() const {
    return m_variableStates;
}
