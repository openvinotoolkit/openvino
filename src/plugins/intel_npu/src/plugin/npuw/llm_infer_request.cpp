// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_infer_request.hpp"

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

    size_t diff_dim = position_shape.size() - 1;
    for (size_t i = 0; i < diff_dim; ++i) {
        OPENVINO_ASSERT(padded_shape[i] == position_shape[i]);
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

void copy_to_right(const ov::SoPtr<ov::ITensor>& src, const ov::SoPtr<ov::ITensor>& dst) {
    OPENVINO_ASSERT(src->get_byte_size() <= dst->get_byte_size());
    std::copy_n(reinterpret_cast<uint8_t*>(src->data()),
                src->get_byte_size(),
                reinterpret_cast<uint8_t*>(dst->data()) + dst->get_byte_size() - src->get_byte_size());
}

void fill_sliding_mask(const ov::SoPtr<ov::ITensor>& mask, int64_t curr_pos, int64_t window_size) {
    auto start = curr_pos - window_size;
    auto end = curr_pos;

    auto* mask_data = mask->data<bool>();
    for (int64_t i = 0; i < static_cast<int64_t>(mask->get_size()); ++i) {
        // Unlike original subgraph which do i <= end we are excluding end
        // as it is a new token and is located in last position of mask buffer
        mask_data[i] = i > start && i < end;
    }

    mask_data[mask->get_size() - 1] = true;
}

constexpr uint32_t INPUT_IDS_SEQ_LEN_DIM = 1;

constexpr std::size_t kStartOutputKVCacheLayers = 1;

}  // anonymous namespace

void ov::npuw::LLMInferRequest::init_lora_states() {
    for (const auto& input_port : m_prefill_request->get_compiled_model()->inputs()) {
        auto input_name = input_port.get_any_name();
        if (ov::npuw::util::matchLoRAMatMulAString(input_name) || ov::npuw::util::matchLoRAMatMulBString(input_name) ||
            ov::npuw::util::matchLoRAMatMulAlphaString(input_name)) {
            auto input_tensor = m_prefill_request->get_tensor(input_port);
            m_variableStates.push_back(std::make_shared<VariableState>(input_name, input_tensor));
        }
    }
}

ov::npuw::LLMInferRequest::LLMInferRequest(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_npuw_llm_compiled_model(compiled_model) {
    for (const auto& input_port : m_npuw_llm_compiled_model->inputs()) {
        init_tensor(input_port);
    }
    for (const auto& output_port : m_npuw_llm_compiled_model->outputs()) {
        init_tensor(output_port);
    }

    auto input_ids_port = find_port_by_name(compiled_model->m_prefill_compiled->inputs(), layer_names::input_ids);
    if (input_ids_port.has_value()) {
        m_input_ids_name = layer_names::input_ids;
    } else {
        OPENVINO_ASSERT(
            find_port_by_name(compiled_model->m_prefill_compiled->inputs(), layer_names::inputs_embeds).has_value());
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

    init_lora_states();

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

    m_generate_initialized = false;
    m_gemma_sliding_window_size = compiled_model->m_gemma_sliding_window_size;
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
    const auto& prefill_compiled = m_npuw_llm_compiled_model->m_prefill_compiled;
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

            auto prefill_lora_in_tensor = m_prefill_request->get_tensor(m_prefill_in_ports.at(state_name));
            auto kvcach_lora_in_tensor = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(state_name));

            // Disable adapter by setting alpha to 0
            if (ov::npuw::util::matchLoRAMatMulAlphaString(state_name)) {
                fill_tensor<float>(prefill_lora_in_tensor, 0.0f);
                fill_tensor<float>(kvcach_lora_in_tensor, 0.0f);
            }
        } else {
            // Generate with LoRA
            auto infer_tensor_shape = m_prefill_request->get_tensor(m_prefill_in_ports.at(state_name))->get_shape();
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

            auto prefill_lora_in_tensor = m_prefill_request->get_tensor(m_prefill_in_ports.at(state_name));
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
            m_prefill_request->set_tensor(m_prefill_in_ports.at(state_name), new_infer_tensor);
            m_kvcache_request->set_tensor(m_kvcache_in_ports.at(state_name), new_infer_tensor);
        }
        variableState->clear_state_updated();
    }
}

void ov::npuw::LLMInferRequest::prepare_for_new_conversation() {
    fill_tensor_bytes(m_prefill_request->get_tensor(m_prefill_in_ports.at(m_input_ids_name)), 0u);
    if (auto type_ids_port = m_prefill_in_ports.find(layer_names::token_type_ids);
        type_ids_port != m_prefill_in_ports.end()) {
        fill_tensor_bytes(m_prefill_request->get_tensor(type_ids_port->second), 0u);
    }
    fill_tensor<int64_t>(m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::attention_mask)), 0);
    fill_tensor<int64_t>(m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::position_ids)), 0);
    m_npuw_llm_compiled_model->m_kvcache_desc.num_stored_tokens = 0u;

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
                    make_tensor_slice(prefill_past_kv, kv_dim, 0u, static_cast<uint32_t>(tokens_in_past_chunks));

                auto kvcache_past_kv_chunks =
                    make_tensor_slice(kvcache_in_tensor, kv_dim, 0u, static_cast<uint32_t>(tokens_in_past_chunks));

                copy_tensor_by_dim(prefill_past_kv_chunks, kvcache_past_kv_chunks, kv_dim);
            }

            // Copy part 2 KV results
            auto prefill_present_kv_chunk =
                make_tensor_slice(prefill_out_tensor,
                                  kv_dim,
                                  static_cast<uint32_t>(prefill_chunk_size - tokens_in_present_chunk),
                                  static_cast<uint32_t>(prefill_chunk_size));

            auto kvcache_last_kv_chunk = make_tensor_slice(kvcache_in_tensor,
                                                           kv_dim,
                                                           static_cast<uint32_t>(tokens_in_past_chunks),
                                                           kvcache_desc.num_stored_tokens);

            copy_tensor_by_dim(prefill_present_kv_chunk, kvcache_last_kv_chunk, kv_dim);
        } else {
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
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports,
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

        // NOTE: Sometimes present kv layer can contain greater seq_len
        //       than was sent to be processed
        uint32_t src_seq_len = static_cast<uint32_t>(src_tensor->get_shape()[kv_dim]);
        OPENVINO_ASSERT(num_tokens <= src_seq_len);
        if (src_seq_len > num_tokens) {
            auto src_slice = make_tensor_slice(src_tensor, kv_dim, src_seq_len - num_tokens, src_seq_len);
            copy_tensor_by_dim(src_slice, dst_slice, kv_dim);
        } else {
            copy_tensor_by_dim(src_tensor, dst_slice, kv_dim);
        }
    }
    LOG_DEBUG("Done.");
}

void ov::npuw::LLMInferRequest::trim_kvcache_for_speculative_decoding(ov::SoPtr<ov::ITensor> position_ids) {
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    // FIXME: It can not work with OmniThinker for now.
    OPENVINO_ASSERT((position_ids->get_shape().size() >= 2) && (position_ids->get_shape().back() >= 1));
    auto position_id = position_ids->data<int64_t>()[0];
    auto dirty_num = kvcache_desc.num_stored_tokens - static_cast<uint32_t>(position_id);
    if (dirty_num > 0) {
        LOG_DEBUG("Trim kv cache from " << kvcache_desc.num_stored_tokens << " length"
                                        << " to " << position_id << " length");
    }
    kvcache_desc.num_stored_tokens -= dirty_num;
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

        fill_tensor<ov::float16>(chunk_prefill_kvcache_in_tensor, 0);
    }
}

void ov::npuw::LLMInferRequest::infer_chunked_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                                      ov::SoPtr<ov::ITensor> attention_mask,
                                                      ov::SoPtr<ov::ITensor> position_ids) {
    LOG_DEBUG("Calling chunked inference for prefill model.");
    LOG_BLOCK();

    const auto input_prompt_len = input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM];

    // For LLM, model accepts 2d inputs_embeds[BATCH, SEQ_LEN]
    // For VLM, model accepts 3d inputs_ids[BATCH, SEQ_LEN, EMB_SIZE]
    bool is_input_embeds = input_ids->get_shape().size() == 2 ? false : true;

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

        std::copy_n(reinterpret_cast<uint8_t*>(input_ids->data()) + prefilled_bytes,
                    current_prefill_bytes,
                    reinterpret_cast<uint8_t*>(input_ids_in_tensor->data()) + input_ids_in_tensor->get_byte_size() -
                        current_prefill_bytes);

        // NB: Regular LLM uses 2D position_ids [BATCH, SEQ_LEN], Qwen2.5 VL/Omni uses 3D position_ids [3, BATCH,
        // SEQ_LEN]
        // Copy postion ids with considering the 3D position_ids
        auto last_dim = position_ids->get_shape().size() - 1;
        auto actual_position_ids_slice =
            make_tensor_slice(position_ids,
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
                                                    ov::SoPtr<ov::ITensor> position_ids,
                                                    ov::SoPtr<ov::ITensor> token_type_ids) {
    LOG_DEBUG("Calling inference for prefill model in a single launch.");
    LOG_BLOCK();

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

    if (token_type_ids) {
        auto padded_token_type_ids = m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::token_type_ids));

        std::fill_n(reinterpret_cast<uint8_t*>(padded_token_type_ids->data()), token_type_ids->get_byte_size(), 0);
        copy_to_right(token_type_ids, padded_token_type_ids);
    }

    auto padded_position_ids = m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::position_ids));
    pad_position_ids(padded_position_ids, position_ids);

    m_prefill_request->infer();
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    kvcache_desc.num_stored_tokens += static_cast<uint32_t>(input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM]);

    LOG_DEBUG("Done");
}

void ov::npuw::LLMInferRequest::infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                              ov::SoPtr<ov::ITensor> attention_mask,
                                              ov::SoPtr<ov::ITensor> position_ids,
                                              ov::SoPtr<ov::ITensor> token_type_ids) {
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
        OPENVINO_ASSERT(m_gemma_sliding_window_size == 0,
            "Chunking is not implemented for Gemma model family yet. "
            "Please use set NPUW_LLM_PREFILL_HINT to 'STATIC'");
        infer_chunked_prefill(input_ids, attention_mask, position_ids);
    } else {
        infer_whole_prefill(input_ids, attention_mask, position_ids, token_type_ids);
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
                                               ov::SoPtr<ov::ITensor> position_ids,
                                               ov::SoPtr<ov::ITensor> token_type_ids) {
    LOG_DEBUG("Calling inference for generate model...");
    LOG_BLOCK();
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    uint32_t input_tokens_len = static_cast<uint32_t>(input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM]);
    if (input_tokens_len > kvcache_desc.max_generation_token_len) {
        OPENVINO_THROW("Input prompt length is greater than output \"NPUW_LLM_MAX_GENERATION_TOKEN_LEN\": ",
                       kvcache_desc.max_generation_token_len,
                       ".\nPlease adjust it.");
    }

    if (!m_generate_initialized) {
        LOG_DEBUG("Copy kv-cache from prefill to generate model.");
        copy_kvcache();

        LOG_DEBUG("Prepare inputs.");
        fill_tensor_bytes(m_kvcache_request->get_tensor(m_kvcache_in_ports.at(m_input_ids_name)), 0u);
        fill_tensor<int64_t>(m_kvcache_request->get_tensor(m_kvcache_in_ports.at(layer_names::attention_mask)), 0);
        fill_tensor<int64_t>(m_kvcache_request->get_tensor(m_kvcache_in_ports.at(layer_names::position_ids)), 0);
        if (token_type_ids) {
            fill_tensor<int64_t>(m_kvcache_request->get_tensor(m_kvcache_in_ports.at(layer_names::token_type_ids)), 0);
        }
        m_generate_initialized = true;
    }

    // NB: KV-cache is full, further generation is impossible
    if (kvcache_desc.num_stored_tokens + input_tokens_len > kvcache_desc.total_size) {
        OPENVINO_THROW("KV-Cache is full.");
    }

    if (auto sliding_mask_port = m_kvcache_in_ports.find(layer_names::gemma_sliding_mask);
        sliding_mask_port != m_kvcache_in_ports.end()) {
        // TODO: Fill once and update on each iteration instead
        fill_sliding_mask(m_kvcache_request->get_tensor(sliding_mask_port->second),
                          kvcache_desc.num_stored_tokens + input_tokens_len,
                          m_gemma_sliding_window_size);
    }

    // FIXME: these tensors should be shared between the parent & child models
    // NB: input_ids can be either fp32(VLM) or i64(LLM)
    auto kv_input_ids = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(m_input_ids_name));
    // NOTE: As `input_tokens_len` can be less than the value of `max_generation_token_len`, which
    //       input layers of generation model are resized to, then we need to put
    //       `input_tokens_len` prompt to the right of `max_generation_token_len`-sized tensors.
    //       Attention mask should rule out all left unusable space.
    std::copy_n(
        reinterpret_cast<uint8_t*>(input_ids->data()),
        input_ids->get_byte_size(),
        reinterpret_cast<uint8_t*>(kv_input_ids->data()) + kv_input_ids->get_byte_size() - input_ids->get_byte_size());

    if (token_type_ids) {
        auto kv_token_type_ids = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(layer_names::token_type_ids));
        copy_to_right(token_type_ids, kv_token_type_ids);
    }

    // NOTE: Attention mask pattern for generate model requires the set of "1"
    //       units of length of the current prompt on the right (for present
    //       kv layers) and the set of "1" units of number of previously calculated
    //       tokens on the left (for past kv layers).
    auto kv_attn_mask = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(layer_names::attention_mask));
    std::copy_n(attention_mask->data<int64_t>(),
                attention_mask->get_size() - input_tokens_len,
                kv_attn_mask->data<int64_t>());
    if (input_tokens_len < kvcache_desc.max_generation_token_len) {
        std::fill_n(kv_attn_mask->data<int64_t>() + kv_attn_mask->get_size() - kvcache_desc.max_generation_token_len,
                    kvcache_desc.max_generation_token_len - input_tokens_len,
                    0);
    }
    std::fill_n(kv_attn_mask->data<int64_t>() + kv_attn_mask->get_size() - input_tokens_len, input_tokens_len, 1);

    auto kv_pos_ids = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(layer_names::position_ids));
    pad_position_ids(kv_pos_ids, position_ids);

    m_kvcache_request->infer();
    kvcache_desc.num_stored_tokens += input_tokens_len;

    if (m_lm_head_request) {
        LOG_DEBUG("Calling inference for LM head model asynchronously");
        m_lm_head_request->start_async();
        if (kvcache_desc.num_stored_tokens < kvcache_desc.total_size) {
            update_kvcache_for(m_kvcache_request, m_kvcache_in_ports, m_kvcache_out_ports, input_tokens_len);
        }
        m_lm_head_request->wait();
        LOG_DEBUG("Calling inference for LM head model -- done.");

        m_logits = m_lm_head_request->get_tensor(m_lm_head_logits_port);
    } else {
        if (kvcache_desc.num_stored_tokens < kvcache_desc.total_size) {
            update_kvcache_for(m_kvcache_request, m_kvcache_in_ports, m_kvcache_out_ports, input_tokens_len);
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

    auto token_type_ids = ov::npuw::util::TensorPtr();

    if (auto type_ids_port = find_port_by_name(inputs, layer_names::token_type_ids); type_ids_port.has_value()) {
        token_type_ids = get_tensor(type_ids_port.value());
    }

    // NB: For VLM, the "inputs_embeds" contains float values (embeddings)
    OPENVINO_ASSERT(ov::element::f32 == input_ids->get_element_type() ||
                    ov::element::i64 == input_ids->get_element_type());
    OPENVINO_ASSERT(ov::element::i64 == attention_mask->get_element_type());
    OPENVINO_ASSERT(ov::element::i64 == position_ids->get_element_type());

    if (m_first_run) {
        // Most of the models have position_ids->data<int64_t>()[0] == 0 for the first infer
        // But gemma3 has it == 1
        // We need to store original first position id in order to distinguish between prefill and generate stage
        // While in most of the cases we need to do prefill only once, it is not true for chat mode
        // where we need to do prefill on each user input.
        m_first_position_id = position_ids->data<int64_t>()[0];
        m_first_run = false;
    }

    // NB: Check the sequence length provided for input_ids
    //     and start position idx in order to distinguish prefill
    //     and generate stages.
    // Notes for Speculative Decoding:
    // 1. If model is a draft one in speculative decoding setting,
    //    we expect it to be launched for more than 1 token only once,
    //    while all other candidates to be generated consequentively
    //    on previous token output.
    // 2. If model is a main one in speculative decoding setting,
    //    then it can be launched on multiple tokens at every iteration.
    //    The first iteration will take the input prompt of variable
    //    length in range [0, NPUW_LLM_MAX_PROMPT_LEN], while others
    //    will be launched on variable number of candidates in range
    //    [0, NPUW_LLM_MAX_GENERATION_TOKEN_LEN].
    //    NPUW_LLM_MAX_GENERATION_TOKEN_LEN is much lesser than
    //    NPUW_LLM_MAX_PROMPT_LEN. So, for second and next iterations
    //    generate model will be utilized, that is reshaped to take
    //    NPUW_LLM_MAX_GENERATION_TOKEN_LEN tokens and output the same
    //    number of logits.
    // The outcome of two items is that prefill and generate stages
    //    can be safely differentiated by start position id for
    //    both main and draft models.
    if (input_ids->get_shape()[INPUT_IDS_SEQ_LEN_DIM] > 1 && position_ids->data<int64_t>()[0] == m_first_position_id) {
        infer_prefill(input_ids, attention_mask, position_ids, token_type_ids);
    } else {
        trim_kvcache_for_speculative_decoding(position_ids);
        infer_generate(input_ids, attention_mask, position_ids, token_type_ids);
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
