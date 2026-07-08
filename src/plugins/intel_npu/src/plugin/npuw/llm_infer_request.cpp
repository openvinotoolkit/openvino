// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_infer_request.hpp"

#include <algorithm>
#include <cmath>
#include <regex>

#include "infer_request_utils.hpp"
#include "llm_block_kvcache_strategy.hpp"
#include "llm_compiled_model.hpp"
#include "llm_continuous_kvcache_strategy.hpp"
#include "logging.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "util.hpp"

namespace {
using ov::npuw::LLMInferRequest;

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

bool is_nonzero_value(const ov::SoPtr<ov::ITensor>& tensor, size_t idx) {
    switch (tensor->get_element_type()) {
    case ov::element::i64:
        return tensor->data<int64_t>()[idx] != 0;
    case ov::element::i32:
        return tensor->data<int32_t>()[idx] != 0;
    case ov::element::u8:
        return tensor->data<uint8_t>()[idx] != 0;
    case ov::element::boolean:
        return tensor->data<bool>()[idx];
    case ov::element::f32:
        return std::fabs(tensor->data<float>()[idx]) > 0.f;
    case ov::element::f16:
        return tensor->data<ov::float16>()[idx] != ov::float16(0.f);
    default:
        OPENVINO_THROW("Unsupported visual_pos_masks source type: ", tensor->get_element_type());
    }
}

// Counts visual tokens (non-zero mask entries) located strictly before sequence position
// `seq_limit`. Used by chunked prefill to compute the deepstack row offset of a chunk: the
// deepstack rows of a chunk follow those of all earlier chunks (visual-token order).
size_t count_visual_tokens_before(const ov::SoPtr<ov::ITensor>& mask, size_t seq_limit) {
    if (!mask || seq_limit == 0) {
        return 0;
    }
    const size_t seq_len = mask->get_shape().back();
    size_t count = 0;
    for (size_t i = 0; i < mask->get_size(); ++i) {
        if ((i % seq_len) < seq_limit && is_nonzero_value(mask, i)) {
            ++count;
        }
    }
    return count;
}

// Scatters the compact deepstack_visual_embeds tensor (one row per visual token, in
// visual-token order) into the right-aligned static destination at the actual
// visual-token sequence positions described by visual_pos_masks.
//
// After the DeepStack gather/scatter cluster is replaced in the graph by a plain
// dense residual add (see ov::npuw::ReplaceDeepstackScatterWithAdd in
// npuw_transformations/replace_deepstack_scatter_with_add.cpp), the destination tensor
// must hold deepstack values at the visual positions and zeros
// everywhere else, so that "hidden + deepstack" reproduces the original per-position
// injection.
//
//   src  : [num_layers, N, emb]   (N visual tokens, k-th row = k-th visual token)
//   mask : [batch, real_len]      (non-zero at visual-token positions, ascending)
//   dst  : [num_layers, seq, emb] (right-aligned static window)
//
// `src_row_offset` skips the first deepstack rows (visual tokens already handled by earlier
// chunks in chunked prefill); it is 0 for whole prefill. Returns the number of visual tokens
// (deepstack rows) actually scattered, so the caller can advance `src_row_offset` for the next
// chunk.
//
// Real tokens are right-aligned, so a visual token at real-sequence coordinate `c`
// lands at static position `c + (seq - real_len)`.
size_t scatter_deepstack_visual_embeds(const ov::SoPtr<ov::ITensor>& src,
                                       const ov::SoPtr<ov::ITensor>& mask,
                                       const ov::SoPtr<ov::ITensor>& dst,
                                       size_t src_row_offset = 0) {
    OPENVINO_ASSERT(dst);
    std::fill_n(reinterpret_cast<uint8_t*>(dst->data()), dst->get_byte_size(), 0);

    if (!src || !mask) {
        return 0;
    }
    if (src->get_element_type() != dst->get_element_type()) {
        return 0;
    }

    const auto& src_shape = src->get_shape();
    const auto& dst_shape = dst->get_shape();
    if (src_shape.size() != 3u || dst_shape.size() != 3u) {
        return 0;
    }

    const size_t num_layers = src_shape[0];
    const size_t src_seq = src_shape[1];
    const size_t emb = src_shape[2];
    const size_t dst_seq = dst_shape[1];
    if (dst_shape[0] != num_layers || dst_shape[2] != emb) {
        return 0;
    }

    const size_t mask_total = mask->get_size();
    OPENVINO_ASSERT(mask_total <= dst_seq);
    // Real tokens are right-aligned in the static window, i.e. left-padded, so this is the
    // amount of left padding (offset of the first real/masked position in dst).
    const size_t seq_left_pad = dst_seq - mask_total;

    const size_t elem_size = src->get_element_type().size();
    const size_t row_bytes = emb * elem_size;
    const auto* src_ptr = static_cast<const uint8_t*>(src->data());
    auto* dst_ptr = static_cast<uint8_t*>(dst->data());

    // Iterate visual-token positions in ascending order; the k-th non-zero mask entry
    // corresponds to deepstack row (src_row_offset + k).
    size_t k = 0;
    for (size_t linear_idx = 0; linear_idx < mask_total; ++linear_idx) {
        if (!is_nonzero_value(mask, linear_idx)) {
            continue;
        }
        OPENVINO_ASSERT(src_row_offset + k < src_seq, "More visual tokens in mask than rows in deepstack source");
        const size_t dst_pos = linear_idx + seq_left_pad;
        for (size_t l = 0; l < num_layers; ++l) {
            const auto* src_row = src_ptr + (l * src_seq + src_row_offset + k) * row_bytes;
            auto* dst_row = dst_ptr + (l * dst_seq + dst_pos) * row_bytes;
            std::copy_n(src_row, row_bytes, dst_row);
        }
        ++k;
    }
    return k;
}

void process_longrope(const std::shared_ptr<ov::IAsyncInferRequest>& infer_req,
                      const LLMInferRequest::PortsMap& ports,
                      const ov::SoPtr<ov::ITensor>& position_ids) {
    if (auto longrope_port_it = ports.find(LLMInferRequest::layer_names::longrope_input);
        longrope_port_it != ports.end()) {
        auto* pos_ids_data = position_ids->data<int64_t>();
        // assuming position_ids are constantly non-deacreasing.
        // this potentially could be not true. Alternative is to find max value in position_ids
        auto max_pos_id = pos_ids_data[position_ids->get_size() - 1];

        auto longrope_input = infer_req->get_tensor(longrope_port_it->second);
        longrope_input->data<int64_t>()[0] = max_pos_id;
    }
}
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
    : ov::npuw::LLMInferBaseRequest(compiled_model) {
    init_ports();

    auto input_ids_port =
        ov::npuw::util::find_port_by_name(compiled_model->m_prefill_compiled->inputs(), layer_names::input_ids);
    if (input_ids_port.has_value()) {
        m_input_ids_name = layer_names::input_ids;
    } else {
        OPENVINO_ASSERT(
            ov::npuw::util::find_port_by_name(compiled_model->m_prefill_compiled->inputs(), layer_names::inputs_embeds)
                .has_value());
        m_input_ids_name = layer_names::inputs_embeds;
    }

    // Create and register all generate model variants.
    auto register_generate_request = [this](const std::shared_ptr<ov::IAsyncInferRequest>& req) {
        m_generate_requests.push_back(req);
        PortsMap in_ports, out_ports;
        for (const auto& p : req->get_compiled_model()->inputs()) {
            in_ports.emplace(p.get_any_name(), p);
        }
        for (const auto& p : req->get_compiled_model()->outputs()) {
            out_ports.emplace(p.get_any_name(), p);
        }
        m_generate_variant_in_ports.emplace(req, std::move(in_ports));
        m_generate_variant_out_ports.emplace(req, std::move(out_ports));
    };
    const size_t num_variants = compiled_model->m_generate_compiled_variants.size();
    m_generate_requests.reserve(num_variants);
    m_generate_base_requests.reserve(num_variants);
    for (size_t i = 0; i < num_variants; ++i) {
        auto base_req = compiled_model->m_generate_compiled_variants[i]->create_base_infer_request();
        m_generate_base_requests.push_back(base_req);
        register_generate_request(compiled_model->m_generate_compiled_variants[i]->wrap_async_infer_request(base_req));
    }
    // Set default to the largest variant for backward compatibility
    m_kvcache_request = m_generate_requests.back();
    // Set ports to ensure tensors aren't empty during bind_past_kv()
    m_kvcache_in_ports = m_generate_variant_in_ports.at(m_kvcache_request);
    m_kvcache_out_ports = m_generate_variant_out_ports.at(m_kvcache_request);

    m_prefill_base_request = compiled_model->m_prefill_compiled->create_base_infer_request();
    m_prefill_request = compiled_model->m_prefill_compiled->wrap_async_infer_request(m_prefill_base_request);

    for (const auto& input_port : m_prefill_request->get_compiled_model()->inputs()) {
        m_prefill_in_ports.emplace(input_port.get_any_name(), input_port);
    }
    for (const auto& output_port : m_prefill_request->get_compiled_model()->outputs()) {
        m_prefill_out_ports.emplace(output_port.get_any_name(), output_port);
    }

    for (const auto& input_port : m_kvcache_request->get_compiled_model()->inputs()) {
        const auto& all_names = input_port.get_names();
        for (const auto& name : all_names) {
            if (ov::npuw::util::starts_with(name, layer_names::past_key_values)) {
                m_kvcache_past_names.push_back(name);
                break;
            }
            if (ov::npuw::util::starts_with_past_lincache(name)) {
                m_lincache_past_names.push_back(name);
                break;
            }
        }
    }

    m_pre_alloc_device = init_pre_alloc_device();
    m_stored_tokens_state = std::make_shared<ov::npuw::StoredTokensState>();
    init_lora_states();

    m_eagle3_ext.initialize(m_npuw_llm_compiled_model->m_is_eagle, m_prefill_in_ports, m_prefill_out_ports);

    // Instantiate the KV cache strategy and run one-time initialization.
    // on_initialize() requires prefill ports, m_generate_requests, and m_pre_alloc_device
    // to be fully populated, so both steps live here at the end of setup.
    if (m_npuw_llm_compiled_model->m_is_block_kv_cache) {
        m_kvcache_strategy = std::make_unique<LLMBlockKVCacheStrategy>(*this);
    } else {
        m_kvcache_strategy = std::make_unique<LLMContinuousKVCacheStrategy>(*this);
    }
    m_kvcache_strategy->on_initialize();

    if (m_npuw_llm_compiled_model->m_enable_prefix_caching) {
        m_prefix_caching_helper = std::make_unique<PrefixCachingHelper>(*this);
    }

    if (compiled_model->m_lm_head_compiled) {
        m_lm_head_request = compiled_model->m_lm_head_compiled->create_infer_request();
        OPENVINO_ASSERT(m_lm_head_request);
        const ov::Output<const ov::Node> lm_head_embed_port = m_lm_head_request->get_inputs()[0];
        m_lm_head_logits_port = m_lm_head_request->get_outputs()[0];
        m_prefill_request->set_tensor(m_prefill_out_ports.at(layer_names::output_embeds),
                                      m_lm_head_request->get_tensor(lm_head_embed_port));

        // Set output_embeds tensor for all generate variants
        for (auto& generate_req : m_generate_requests) {
            const auto& variant_out_ports = m_generate_variant_out_ports.at(generate_req);
            generate_req->set_tensor(variant_out_ports.at(layer_names::output_embeds),
                                     m_lm_head_request->get_tensor(lm_head_embed_port));
        }
    }

    // FIXME: E-177589
    // FIXME: "fixes"/workarounds caching import on CPU (also might be related to bf16 weights).
    // Unclear how it's related. Previously fill_tensor()
    // was in copy_kvcache() call. When it was removed, it broke the import accuracy.
    bool enable_cpu_wa = false;
    const auto& kvcache_compiled = m_npuw_llm_compiled_model->m_kvcache_compiled;
    for (std::size_t idx = 0; idx < kvcache_compiled->num_submodels(); ++idx) {
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
        // Apply CPU workaround only for the largest variant since all variants share its past KV tensors
        auto& largest_kvcache_req = m_generate_requests.back();
        const auto& variant_in_ports = m_generate_variant_in_ports.at(largest_kvcache_req);
        for (const auto& kv_past_name : m_kvcache_past_names) {
            auto kvcache_in_tensor = largest_kvcache_req->get_tensor(variant_in_ports.at(kv_past_name));
            // NB: Use fill_tensor_bytes to zero-fill regardless of element type.
            // fill_tensor<ov::float16> would fail with a type mismatch error for i8 kv-cache
            ov::npuw::util::fill_tensor_bytes(kvcache_in_tensor, 0u);
        }
        for (const auto& lincache_past_name : m_lincache_past_names) {
            auto lincache_in_tensor = largest_kvcache_req->get_tensor(variant_in_ports.at(lincache_past_name));
            ov::npuw::util::fill_tensor_bytes(lincache_in_tensor, 0u);
        }
    }

    m_generate_initialized = false;

    m_llm_profile.report_on_die = ov::npuw::profiling_enabled();
    m_llm_profile.area = "LLM/execution";
}

std::string ov::npuw::LLMInferRequest::init_pre_alloc_device() {
    bool pre_alloc_on_npu = true;
    const auto& kvcache_compiled = m_npuw_llm_compiled_model->m_kvcache_compiled;
    for (std::size_t idx = 0; idx < kvcache_compiled->num_submodels(); ++idx) {
        if (kvcache_compiled->submodel_device(idx) != "NPU") {
            pre_alloc_on_npu = false;
            break;
        }
    }

    return pre_alloc_on_npu ? "NPU" : "CPU";
}

void ov::npuw::LLMInferRequest::bind_past_kv() {
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    if (kvcache_desc.v_tensors_transposed_pre != kvcache_desc.v_tensors_transposed_gen) {
        // FIXME: disable kv cache sharing when one of the models is transposed for now
        return;
    }

    // Only reuse KV cache related tensors (past_key_values)
    for (const auto& [input_name, prefill_in_port] : m_prefill_in_ports) {
        // Only process KV cache inputs (past_key_values)
        if (input_name.find(layer_names::past_key_values) == std::string::npos) {
            continue;
        }

        // Check if the kv cache request has the same input port
        if (m_kvcache_in_ports.find(input_name) == m_kvcache_in_ports.end()) {
            continue;
        }

        const auto& kvcache_in_port = m_kvcache_in_ports.at(input_name);
        const auto& kvcache_past_kv_in_tensor = m_kvcache_request->get_tensor(kvcache_in_port);
        auto data = kvcache_past_kv_in_tensor->data();

        auto origTensor = m_prefill_request->get_tensor(prefill_in_port);
        auto new_tensor =
            ov::get_tensor_impl(ov::Tensor(origTensor->get_element_type(), origTensor->get_shape(), data));
        m_prefill_request->set_tensor(prefill_in_port, new_tensor);

        // Record that we have already bind past_kv, will need data copy when update past kv in infer requests to
        // ensure correct data layout
        m_past_kv_bound = true;
    }
}

std::shared_ptr<ov::IAsyncInferRequest> ov::npuw::LLMInferRequest::select_generate_request(int64_t prompt_length) {
    // Select the largest variant if prompt_length is 0 (unknown)
    if (prompt_length == 0) {
        LOG_DEBUG("Prompt length unknown, using largest variant");
        return m_generate_requests.back();
    }

    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    // Calculate expected total tokens: prompt + min_response_len
    // min_response_len = total_size - max_prompt_size
    uint32_t min_response_len = kvcache_desc.total_size - kvcache_desc.max_prompt_size;
    int64_t expected_total_tokens = prompt_length + min_response_len;

    const auto& kvcache_sizes = m_npuw_llm_compiled_model->m_kvcache_sizes;
    // Find the smallest variant that can accommodate the expected token count
    for (size_t i = 0; i < kvcache_sizes.size(); ++i) {
        if (expected_total_tokens <= kvcache_sizes[i]) {
            LOG_DEBUG("Selected generate request " << (i + 1) << "/" << kvcache_sizes.size() << " with size "
                                                   << kvcache_sizes[i] << " for prompt_length=" << prompt_length
                                                   << " (expected_total=" << expected_total_tokens << " tokens)");
            return m_generate_requests[i];
        }
    }

    // Fallback to the largest variant if expected_total_tokens exceeds all predefined sizes
    LOG_WARN("No suitable generate request found for expected_total_tokens="
             << expected_total_tokens << " (prompt_length=" << prompt_length << "), using largest variant");
    return m_generate_requests.back();
}

void ov::npuw::LLMInferRequest::apply_lora() {
    uint32_t max_low_rank_dim_size = m_npuw_llm_compiled_model->m_max_lora_rank;

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
                ov::npuw::util::fill_tensor<float>(prefill_lora_in_tensor, 0.0f);
                ov::npuw::util::fill_tensor<float>(kvcach_lora_in_tensor, 0.0f);
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
                                                             m_pre_alloc_device,
                                                             m_npuw_llm_compiled_model->get_plugin());
            bool has_padding = state_tensor_rank != target_lora_rank;
            if (has_padding) {
                // Clear padding tensor in infer request
                ov::npuw::util::fill_tensor<float>(new_infer_tensor, 0.0f);
            }

            // Fill LoRA into infer request
            auto fill_lora_in_tensor = [low_rank_dim, state_tensor_rank](ov::SoPtr<ov::ITensor> state_tensor,
                                                                         ov::SoPtr<ov::ITensor> infer_tensor,
                                                                         bool has_padding) {
                if (!has_padding) {
                    state_tensor->copy_to(infer_tensor._ptr);
                    return;
                }

                auto new_tensor_slice =
                    ov::npuw::util::make_tensor_slice(infer_tensor, low_rank_dim, 0u, state_tensor_rank);
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
    prepare_for_new_conversation(0);
}

void ov::npuw::LLMInferRequest::prepare_for_new_conversation(int64_t prompt_length) {
    namespace uu = ov::npuw::util;

    uu::fill_tensor_bytes(m_prefill_request->get_tensor(m_prefill_in_ports.at(m_input_ids_name)), 0u);
    if (auto type_ids_port = m_prefill_in_ports.find(layer_names::token_type_ids);
        type_ids_port != m_prefill_in_ports.end()) {
        uu::fill_tensor_bytes(m_prefill_request->get_tensor(type_ids_port->second), 0u);
    }
    uu::fill_tensor<int64_t>(m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::attention_mask)), 0);
    uu::fill_tensor<int64_t>(m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::position_ids)), 0);

    // Gemma4: Clear per_layer_inputs if present
    if (auto per_layer_port = m_prefill_in_ports.find(layer_names::per_layer_inputs);
        per_layer_port != m_prefill_in_ports.end()) {
        uu::fill_tensor_bytes(m_prefill_request->get_tensor(per_layer_port->second), 0u);
    }

    m_kvcache_strategy->on_reset(prompt_length > 0 ? static_cast<uint32_t>(prompt_length) : 0u);

    for (const auto& input_name : m_lincache_past_names) {
        if (m_prefill_in_ports.find(input_name) != m_prefill_in_ports.end()) {
            uu::fill_tensor_bytes(m_prefill_request->get_tensor(m_prefill_in_ports.at(input_name)), 0u);
        }
    }

    m_npuw_llm_compiled_model->m_kvcache_desc.num_stored_tokens = 0u;

    // Select the appropriate generate inference request variant based on prompt length
    // The function internally calculates expected total tokens (prompt + min_response_len)
    m_kvcache_request = select_generate_request(prompt_length);
    m_kvcache_in_ports = m_generate_variant_in_ports.at(m_kvcache_request);
    m_kvcache_out_ports = m_generate_variant_out_ports.at(m_kvcache_request);
}

void ov::npuw::LLMInferRequest::copy_kvcache() {
    namespace uu = ov::npuw::util;
    LOG_DEBUG("Copying kv-cache from prefill to generate model.");
    LOG_BLOCK();
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    // FIXME: Find only matching by names outputs and copy them, having previously checked that such inputs exist
    ov::parallel_for(m_kvcache_past_names.size(), [&](size_t out_idx) {
        const auto& input_name = m_kvcache_past_names[out_idx];
        auto kvcache_in_tensor = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(input_name));

        const auto& output_name = std::regex_replace(input_name, std::regex(layer_names::past_key_values), "present");
        OPENVINO_ASSERT(m_prefill_out_ports.find(output_name) != m_prefill_out_ports.end(),
                        "Incosistent input/output naming for KV cache: ",
                        output_name,
                        " not found in prefill model outputs.");
        auto prefill_out_tensor = m_prefill_request->get_tensor(m_prefill_out_ports.at(output_name));

        const auto is_value_tensor = output_name.find("value") != std::string::npos;
        const auto kv_dim = [&](bool v_trans) -> uint32_t {
            return (is_value_tensor && v_trans) ? 3u : kvcache_desc.dim;
        };

        const auto& pre_kv_dim = kv_dim(kvcache_desc.v_tensors_transposed_pre);
        const auto& gen_kv_dim = kv_dim(kvcache_desc.v_tensors_transposed_gen);

        const auto prefill_chunk_size = m_npuw_llm_compiled_model->m_prefill_chunk_size;
        const bool use_chunk_prefill = m_npuw_llm_compiled_model->m_use_chunk_prefill;
        if (use_chunk_prefill) {
            // The chunk prefilled KV results are divided into two parts:
            // Part 1: The KV results from loops 1 to n-1 have been copied into the 'past' KV input tensor
            // Part 2: The kv results from the last loop remain in the 'present' KV output tensor
            // The task is to copy both parts into the KV-cache input tensor for the decoding process

            // Copy part 1 KV results
            // tokens_in_past_chunks may be 0 in case short prompts are prefilled in single chunk
            auto tokens_in_past_chunks = kvcache_desc.num_stored_tokens - m_tokens_in_present_chunk;
            if (tokens_in_past_chunks > 0) {
                // Create backup of past KV tensor when buffer sharing is enabled to prevent data corruption
                // This is necessary because subsequent copy operations would overwrite the shared buffer
                auto prefill_past_kv = m_prefill_request->get_tensor(m_prefill_in_ports.at(input_name));
                ov::SoPtr<ov::ITensor> tmp_dense_kv_tensor;
                ov::SoPtr<ov::ITensor> prefill_past_kv_chunks;
                if (m_past_kv_bound) {
                    tmp_dense_kv_tensor = ov::npuw::util::allocMem(prefill_past_kv->get_element_type(),
                                                                   prefill_past_kv->get_shape(),
                                                                   m_pre_alloc_device,
                                                                   m_npuw_llm_compiled_model->get_plugin());
                    prefill_past_kv->copy_to(tmp_dense_kv_tensor._ptr);
                    prefill_past_kv_chunks = make_tensor_slice(tmp_dense_kv_tensor,
                                                               pre_kv_dim,
                                                               0u,
                                                               static_cast<uint32_t>(tokens_in_past_chunks));
                } else {
                    prefill_past_kv_chunks = make_tensor_slice(prefill_past_kv,
                                                               pre_kv_dim,
                                                               0u,
                                                               static_cast<uint32_t>(tokens_in_past_chunks));
                }

                auto kvcache_past_kv_chunks = uu::make_tensor_slice(kvcache_in_tensor,
                                                                    gen_kv_dim,
                                                                    0u,
                                                                    static_cast<uint32_t>(tokens_in_past_chunks));

                uu::copy_tensor_by_dim(prefill_past_kv_chunks, kvcache_past_kv_chunks, pre_kv_dim, gen_kv_dim);
            }

            // Copy part 2 KV results
            auto prefill_present_kv_chunk =
                uu::make_tensor_slice(prefill_out_tensor,
                                      pre_kv_dim,
                                      static_cast<uint32_t>(prefill_chunk_size - m_tokens_in_present_chunk),
                                      static_cast<uint32_t>(prefill_chunk_size));

            auto kvcache_last_kv_chunk = uu::make_tensor_slice(kvcache_in_tensor,
                                                               gen_kv_dim,
                                                               static_cast<uint32_t>(tokens_in_past_chunks),
                                                               kvcache_desc.num_stored_tokens);

            uu::copy_tensor_by_dim(prefill_present_kv_chunk, kvcache_last_kv_chunk, pre_kv_dim, gen_kv_dim);
        } else {
            auto prefill_out_slice =
                uu::make_tensor_slice(prefill_out_tensor,
                                      pre_kv_dim,
                                      kvcache_desc.max_prompt_size - kvcache_desc.num_stored_tokens,
                                      kvcache_desc.max_prompt_size);

            auto kvcache_in_slice =
                uu::make_tensor_slice(kvcache_in_tensor, gen_kv_dim, 0u, kvcache_desc.num_stored_tokens);

            uu::copy_tensor_by_dim(prefill_out_slice, kvcache_in_slice, pre_kv_dim, gen_kv_dim);
        }
    });
    LOG_DEBUG("Done.");
}

void ov::npuw::LLMInferRequest::update_kvcache_for(
    std::shared_ptr<ov::IAsyncInferRequest> request,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports,
    uint32_t num_tokens,
    bool v_transposed) {
    namespace uu = ov::npuw::util;
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;

    for (std::size_t i = 0; i < m_kvcache_past_names.size(); ++i) {
        const auto& input_name = m_kvcache_past_names[i];
        OPENVINO_ASSERT(in_ports.find(input_name) != in_ports.end(),
                        "There is no ",
                        input_name,
                        " in input ports map, while it is expected!");
        const auto& output_name = std::regex_replace(input_name, std::regex(layer_names::past_key_values), "present");
        OPENVINO_ASSERT(out_ports.find(output_name) != out_ports.end(),
                        "There is no ",
                        output_name,
                        " in output ports map, while it is expected!");

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
}

void ov::npuw::LLMInferRequest::copy_lincache(
    std::shared_ptr<ov::IAsyncInferRequest> from_request,
    std::shared_ptr<ov::IAsyncInferRequest> to_request,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& from_ports,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& to_ports) {
    namespace uu = ov::npuw::util;
    LOG_DEBUG("Copying linear cache.");
    LOG_BLOCK();
    ov::parallel_for(m_lincache_past_names.size(), [&](size_t out_idx) {
        const auto& input_name = m_lincache_past_names[out_idx];
        OPENVINO_ASSERT(to_ports.find(input_name) != to_ports.end(),
                        "Incosistent input/output naming for linear cache: ",
                        input_name,
                        " not found in model inputs.");
        auto to_tensor = to_request->get_tensor(to_ports.at(input_name));

        const auto& output_name = std::regex_replace(input_name, std::regex("past"), "present");
        OPENVINO_ASSERT(from_ports.find(output_name) != from_ports.end(),
                        "Incosistent input/output naming for linear cache: ",
                        output_name,
                        " not found in model outputs.");
        auto from_tensor = from_request->get_tensor(from_ports.at(output_name));

        from_tensor->copy_to(to_tensor._ptr);
    });
    LOG_DEBUG("Done.");
}

void ov::npuw::LLMInferRequest::trim_kvcache_for_speculative_decoding(ov::SoPtr<ov::ITensor> position_ids) {
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    // FIXME: It won't work with Qwen2.5-VL/Omni for now.
    OPENVINO_ASSERT((position_ids->get_shape().size() == 2) && (position_ids->get_shape().back() >= 1));
    auto position_id = position_ids->data<int64_t>()[0];
    auto dirty_num = kvcache_desc.num_stored_tokens - static_cast<uint32_t>(position_id);
    if (dirty_num > 0) {
        LOG_DEBUG("Trim kv cache from " << kvcache_desc.num_stored_tokens << " length" << " to " << position_id
                                        << " length");
    }
    kvcache_desc.num_stored_tokens -= dirty_num;
}

void ov::npuw::LLMInferRequest::clear_chunk_prefill_kv_cache() {
    for (const auto& input_name : m_kvcache_past_names) {
        OPENVINO_ASSERT(m_prefill_in_ports.find(input_name) != m_prefill_in_ports.end(),
                        "Incosistent input/output naming for KV cache: ",
                        input_name,
                        " not found in prefill model inputs.");

        auto chunk_prefill_kvcache_in_tensor = m_prefill_request->get_tensor(m_prefill_in_ports.at(input_name));

        // NB: Use fill_tensor_bytes to zero-fill regardless of element type.
        // After KV cache compression, past KV inputs may be i8 precision, and
        // fill_tensor<ov::float16> would fail with a type mismatch error.
        ov::npuw::util::fill_tensor_bytes(chunk_prefill_kvcache_in_tensor, 0u);
    }
}

void ov::npuw::LLMInferRequest::infer_chunked_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                                      ov::SoPtr<ov::ITensor> attention_mask,
                                                      ov::SoPtr<ov::ITensor> position_ids,
                                                      ov::SoPtr<ov::ITensor> per_layer_inputs,
                                                      ov::SoPtr<ov::ITensor> visual_pos_masks,
                                                      ov::SoPtr<ov::ITensor> deepstack_visual_embeds) {
    LOG_DEBUG("Calling chunked inference for prefill model.");
    LOG_BLOCK();

    const auto input_prompt_len = input_ids->get_shape()[layer_ids::INPUT_IDS_SEQ_LEN_DIM];

    // For LLM, model accepts 2d inputs_ids[BATCH, SEQ_LEN]
    // For VLM, model accepts 3d inputs_embeds[BATCH, SEQ_LEN, EMB_SIZE]
    bool is_input_embeds = input_ids->get_shape().size() == 2 ? false : true;

    const auto input_ids_elem_size = input_ids->get_element_type().size();
    auto input_ids_in_tensor = m_prefill_request->get_tensor(m_prefill_in_ports.at(m_input_ids_name));
    const uint64_t chunk_prompt_len = m_npuw_llm_compiled_model->m_prefill_chunk_size;

    // DeepStack (Qwen3-VL): the deepstack injection is scattered per chunk inside the loop
    // below, the same way attention_mask / input_ids are sliced for the current chunk.
    const auto deepstack_it = m_prefill_in_ports.find(layer_names::deepstack_visual_embeds);
    const bool has_deepstack = deepstack_it != m_prefill_in_ports.end();
    if (has_deepstack) {
        OPENVINO_ASSERT(deepstack_visual_embeds && visual_pos_masks,
                        "deepstack_visual_embeds and visual_pos_masks must be provided for DeepStack VLM prefill.");
    }

    auto attn_mask_in_tensor = m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::attention_mask));
    auto pos_ids_in_tensor = m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::position_ids));

    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;

    uint64_t remaining_prompts = input_prompt_len;

    const bool enable_prefix_caching = m_npuw_llm_compiled_model->m_enable_prefix_caching;
    PrefixCacheRestorationContext cache_context;
    if (enable_prefix_caching) {
        // Prepare and restore prefix cache using helper
        cache_context = m_prefix_caching_helper->prepare_and_restore(input_ids, input_prompt_len);
        remaining_prompts = cache_context.remaining_prompts;
    }

    if (m_eagle3_ext.is_eagle3_model()) {
        m_eagle3_ext.reset_chunked_prefill_state();
    }

    // DeepStack rows are consumed in visual-token order across chunks; track how many have
    // already been scattered so each chunk continues where the previous one left off.
    size_t visual_tokens_scattered =
        has_deepstack ? count_visual_tokens_before(visual_pos_masks, kvcache_desc.num_stored_tokens) : 0u;

    while (remaining_prompts > 0) {
        // NB: input_ids can be either fp32(VLM) or i64(LLM)
        // The last chunk may not be completely filled if the actual length of the prompts is not evenly divisible by
        // the chunk size
        auto current_prompts_len = std::min(remaining_prompts, chunk_prompt_len);

        m_llm_profile["1/prefill:3a.prepare_chunk"].record([&]() {
            // Handle first chunk with prefix caching: populate attention mask for restored cache
            if (enable_prefix_caching && cache_context.restore_prefix_cache) {
                m_prefix_caching_helper->populate_attention_mask_for_restored_cache(attention_mask,
                                                                                    attn_mask_in_tensor,
                                                                                    kvcache_desc.num_stored_tokens);
                cache_context.restore_prefix_cache = false;
            }

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
            if (is_input_embeds) {
                current_prefill_bytes *= input_ids->get_shape().back();
                prefilled_bytes *= input_ids->get_shape().back();
            }

            ov::npuw::util::fill_tensor_bytes(input_ids_in_tensor, 0u);
            std::copy_n(reinterpret_cast<uint8_t*>(input_ids->data()) + prefilled_bytes,
                        current_prefill_bytes,
                        reinterpret_cast<uint8_t*>(input_ids_in_tensor->data()) + input_ids_in_tensor->get_byte_size() -
                            current_prefill_bytes);

            // NB: Regular LLM uses 2D position_ids [BATCH, SEQ_LEN], Qwen2.5 VL/Omni, Qwen3.5 VL use 3D position_ids
            // [3, BATCH, SEQ_LEN]
            // Copy postion ids with considering the 3D position_ids
            auto last_dim = position_ids->get_shape().size() - 1;
            auto actual_position_ids_slice = ov::npuw::util::make_tensor_slice(
                position_ids,
                static_cast<uint32_t>(last_dim),
                kvcache_desc.num_stored_tokens,
                kvcache_desc.num_stored_tokens + static_cast<uint32_t>(current_prompts_len));

            auto pos_ids_slice =
                ov::npuw::util::make_tensor_slice(pos_ids_in_tensor,
                                                  static_cast<uint32_t>(last_dim),
                                                  static_cast<uint32_t>(chunk_prompt_len - current_prompts_len),
                                                  static_cast<uint32_t>(chunk_prompt_len));

            // Copy with proper stride handling
            actual_position_ids_slice->copy_to(pos_ids_slice._ptr);

            // DeepStack (Qwen3-VL): scatter only the visual tokens that fall into the current
            // chunk. The chunk's visual_pos_masks slice gives their chunk-local positions, and
            // the deepstack rows for those tokens follow the visual tokens of earlier chunks.
            if (has_deepstack) {
                auto deepstack_local = m_prefill_request->get_tensor(deepstack_it->second);
                const uint32_t seq_dim = static_cast<uint32_t>(visual_pos_masks->get_shape().size() - 1);
                auto chunk_mask = ov::npuw::util::make_tensor_slice(
                    visual_pos_masks,
                    seq_dim,
                    static_cast<uint32_t>(kvcache_desc.num_stored_tokens),
                    static_cast<uint32_t>(kvcache_desc.num_stored_tokens + current_prompts_len));
                visual_tokens_scattered += scatter_deepstack_visual_embeds(deepstack_visual_embeds,
                                                                           chunk_mask._ptr,
                                                                           deepstack_local,
                                                                           visual_tokens_scattered);
            }

            if (m_eagle3_ext.is_eagle3_model()) {
                m_eagle3_ext.prepare_inputs_for_chunk(m_prefill_request,
                                                      m_prefill_in_ports,
                                                      kvcache_desc.num_stored_tokens,
                                                      static_cast<uint32_t>(current_prompts_len));
            }

            // Update history size for dynamic context:
            // dynamic attention selector needs history size to determin the past KV shape and attention mask shape
            m_prefill_base_request->update_history_size(kvcache_desc.num_stored_tokens);

            // Gemma4: copy the current chunk of per_layer_inputs right-aligned on seq_len dim.
            // Source shape: [1, input_prompt_len, num_layers, proj_dim]
            // Dest shape:   [1, chunk_prompt_len, num_layers, proj_dim] (static)
            if (per_layer_inputs) {
                auto dst = m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::per_layer_inputs));
                ov::npuw::util::copy_per_layer_inputs_chunk_to_right(per_layer_inputs,
                                                                     dst,
                                                                     kvcache_desc.num_stored_tokens,
                                                                     static_cast<uint32_t>(current_prompts_len));
            }

            // Prepare KV blocks or bind memory for this chunk via strategy.
            m_kvcache_strategy->on_prefill_chunk_begin(static_cast<uint32_t>(current_prompts_len));
        });

        m_llm_profile["1/prefill:3b.infer"].record([&]() {
            m_prefill_request->infer();
        });

        m_llm_profile["1/prefill:3c.post_chunk"].record([&]() {
            // Accumulate Eagle3 last_hidden_state from this chunk
            if (m_eagle3_ext.is_eagle3_model()) {
                m_eagle3_ext.accumulate_chunk_last_hidden_state(m_prefill_request,
                                                                m_prefill_out_ports,
                                                                static_cast<uint32_t>(current_prompts_len),
                                                                static_cast<uint32_t>(input_prompt_len));
            }

            if (enable_prefix_caching) {
                m_prefix_caching_helper->store_computed_blocks(current_prompts_len,
                                                               cache_context.prompt_hashes,
                                                               cache_context.token_idx);
            }
        });

        const bool is_last_chunk = (remaining_prompts - current_prompts_len) <= 0;
        remaining_prompts -= current_prompts_len;
        kvcache_desc.num_stored_tokens += static_cast<uint32_t>(current_prompts_len);

        m_llm_profile["1/prefill:3d.update_kvcache"].record([&]() {
            // Finalise KV state after inference via strategy.
            m_kvcache_strategy->on_prefill_chunk_done(static_cast<uint32_t>(current_prompts_len), is_last_chunk);
            // Do not copy last computed chunk and preserve it in present k/v layer.
            if (!is_last_chunk) {
                // Attention mask and lincache update for intermediate chunks.
                copy_lincache(m_prefill_request, m_prefill_request, m_prefill_out_ports, m_prefill_in_ports);

                std::copy_n(
                    attn_mask_in_tensor->data<int64_t>() + attn_mask_in_tensor->get_size() - current_prompts_len,
                    current_prompts_len,
                    attn_mask_in_tensor->data<int64_t>() + kvcache_desc.num_stored_tokens - current_prompts_len);
            }
        });

        if (is_last_chunk) {
            LOG_DEBUG("All prompts have been prefilled in chunks");
            m_tokens_in_present_chunk = current_prompts_len;
            break;
        }
    }

    LOG_DEBUG("Done.");

    if (enable_prefix_caching) {
        m_prefix_caching_helper->print_cache_status();
    }
}

void ov::npuw::LLMInferRequest::infer_whole_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                                    ov::SoPtr<ov::ITensor> attention_mask,
                                                    ov::SoPtr<ov::ITensor> position_ids,
                                                    ov::SoPtr<ov::ITensor> token_type_ids,
                                                    ov::SoPtr<ov::ITensor> per_layer_inputs,
                                                    ov::SoPtr<ov::ITensor> visual_pos_masks,
                                                    ov::SoPtr<ov::ITensor> deepstack_visual_embeds) {
    LOG_DEBUG("Calling inference for prefill model in a single launch.");
    LOG_BLOCK();

    m_llm_profile["1/prefill:3a.prepare"].record([&]() {
        // NB: padded_input can be either fp32(VLM) or i64(LLM)
        auto padded_input = m_prefill_request->get_tensor(m_prefill_in_ports.at(m_input_ids_name));
        std::copy_n(reinterpret_cast<uint8_t*>(input_ids->data()),
                    input_ids->get_byte_size(),
                    reinterpret_cast<uint8_t*>(padded_input->data()) + padded_input->get_byte_size() -
                        input_ids->get_byte_size());

        auto padded_attention_mask = m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::attention_mask));
        std::copy_n(
            attention_mask->data<int64_t>(),
            attention_mask->get_size(),
            padded_attention_mask->data<int64_t>() + padded_attention_mask->get_size() - attention_mask->get_size());

        if (token_type_ids) {
            auto padded_token_type_ids =
                m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::token_type_ids));

            std::fill_n(reinterpret_cast<uint8_t*>(padded_token_type_ids->data()), token_type_ids->get_byte_size(), 0);
            util::copy_to_right(token_type_ids, padded_token_type_ids);
        }

        auto padded_position_ids = m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::position_ids));
        ov::npuw::util::pad_position_ids(padded_position_ids, position_ids);

        if (const auto deepstack_it = m_prefill_in_ports.find(layer_names::deepstack_visual_embeds);
            deepstack_it != m_prefill_in_ports.end()) {
            OPENVINO_ASSERT(deepstack_visual_embeds && visual_pos_masks,
                            "deepstack_visual_embeds and visual_pos_masks must be provided for DeepStack VLM prefill.");
            auto deepstack_local = m_prefill_request->get_tensor(deepstack_it->second);
            scatter_deepstack_visual_embeds(deepstack_visual_embeds, visual_pos_masks, deepstack_local);
        }

        if (m_eagle3_ext.is_eagle3_model()) {
            m_eagle3_ext.prepare_inputs(m_prefill_request, m_prefill_in_ports);
        }

        // Gemma4: pass per_layer_inputs from external inputs into the prefill sub-request.
        // Shape is [1, seq_len, num_layers, proj_dim]; copy right-aligned on the seq_len dim.
        if (per_layer_inputs) {
            auto dst = m_prefill_request->get_tensor(m_prefill_in_ports.at(layer_names::per_layer_inputs));
            ov::npuw::util::copy_to_right(per_layer_inputs, dst);
        }
    });

    m_llm_profile["1/prefill:3b.infer"].record([&]() {
        m_prefill_request->infer();
    });

    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    kvcache_desc.num_stored_tokens += static_cast<uint32_t>(input_ids->get_shape()[layer_ids::INPUT_IDS_SEQ_LEN_DIM]);

    LOG_DEBUG("Done");
}

void ov::npuw::LLMInferRequest::infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                                              ov::SoPtr<ov::ITensor> attention_mask,
                                              ov::SoPtr<ov::ITensor> position_ids,
                                              ov::SoPtr<ov::ITensor> token_type_ids,
                                              ov::SoPtr<ov::ITensor> per_layer_inputs,
                                              ov::SoPtr<ov::ITensor> visual_pos_masks,
                                              ov::SoPtr<ov::ITensor> deepstack_visual_embeds) {
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

    m_llm_profile["1/prefill:1.prepare_for_new_conversation"].record([&]() {
        prepare_for_new_conversation(prompt_length);
    });

    process_longrope(m_prefill_request, m_prefill_in_ports, position_ids);

    m_llm_profile["1/prefill:2.apply_lora"].record([&]() {
        apply_lora();
    });

    const bool use_chunk_prefill = m_npuw_llm_compiled_model->m_use_chunk_prefill;
    m_llm_profile["1/prefill:3.infer"].record([&]() {
        if (use_chunk_prefill) {
            OPENVINO_ASSERT(!token_type_ids,
                            "Chunking is not implemented for Gemma model family yet. "
                            "Please set NPUW_LLM_PREFILL_HINT to 'STATIC'");
            infer_chunked_prefill(input_ids,
                                  attention_mask,
                                  position_ids,
                                  per_layer_inputs,
                                  visual_pos_masks,
                                  deepstack_visual_embeds);
        } else {
            infer_whole_prefill(input_ids,
                                attention_mask,
                                position_ids,
                                token_type_ids,
                                per_layer_inputs,
                                visual_pos_masks,
                                deepstack_visual_embeds);
        }
    });

    m_llm_profile["1/prefill:4.lm_head"].record([&]() {
        if (m_lm_head_request) {
            LOG_DEBUG("Calling inference for LM head model.");
            m_lm_head_request->infer();
            m_logits = m_lm_head_request->get_tensor(m_lm_head_logits_port);
        } else {
            m_logits = m_prefill_request->get_tensor(m_prefill_out_ports.at(layer_names::logits));
        }

        // Update last_hidden_state only for non-chunked prefill
        // For chunked prefill, accumulate_chunk_last_hidden_state() already set the tensor
        if (m_eagle3_ext.is_eagle3_model() && !use_chunk_prefill) {
            m_eagle3_ext.update_last_hidden_state(m_prefill_request, m_prefill_out_ports);
        }
    });

    m_generate_initialized = false;

    LOG_DEBUG("Done");
}

void ov::npuw::LLMInferRequest::infer_generate(ov::SoPtr<ov::ITensor> input_ids,
                                               ov::SoPtr<ov::ITensor> attention_mask,
                                               ov::SoPtr<ov::ITensor> position_ids,
                                               ov::SoPtr<ov::ITensor> token_type_ids,
                                               ov::SoPtr<ov::ITensor> per_layer_inputs) {
    LOG_DEBUG("Calling inference for generate model...");
    LOG_BLOCK();
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    uint32_t input_tokens_len = static_cast<uint32_t>(input_ids->get_shape()[layer_ids::INPUT_IDS_SEQ_LEN_DIM]);
    if (input_tokens_len > kvcache_desc.max_generation_token_len) {
        OPENVINO_THROW("Input prompt length is greater than output \"NPUW_LLM_MAX_GENERATION_TOKEN_LEN\": ",
                       kvcache_desc.max_generation_token_len,
                       ".\nPlease adjust it.");
    }

    // Note: m_kvcache_request, m_kvcache_in_ports, and m_kvcache_out_ports are selected in
    // prepare_for_new_conversation()

    // Eagle3: Check for sampling result from external pipeline via VariableState
    if (m_eagle3_ext.is_eagle3_model() && m_generate_initialized) {
        m_eagle3_ext.process_sampling_result_from_state(m_kvcache_request,
                                                        m_kvcache_in_ports,
                                                        kvcache_desc.num_stored_tokens,
                                                        kvcache_desc.v_tensors_transposed_gen,
                                                        kvcache_desc.dim);
    }

    m_llm_profile["N/generate:1.prepare"].record([&]() {
        if (!m_generate_initialized) {
            LOG_DEBUG("Copy kv-cache from prefill to generate model.");
            if (kvcache_desc.num_stored_tokens > 0) {
                m_kvcache_strategy->on_generate_kv_init();
                copy_lincache(m_prefill_request, m_kvcache_request, m_prefill_out_ports, m_kvcache_in_ports);
            }

            LOG_DEBUG("Prepare inputs.");
            namespace uu = ov::npuw::util;
            uu::fill_tensor_bytes(m_kvcache_request->get_tensor(m_kvcache_in_ports.at(m_input_ids_name)), 0u);
            uu::fill_tensor<int64_t>(m_kvcache_request->get_tensor(m_kvcache_in_ports.at(layer_names::attention_mask)),
                                     0);
            uu::fill_tensor<int64_t>(m_kvcache_request->get_tensor(m_kvcache_in_ports.at(layer_names::position_ids)),
                                     0);
            if (token_type_ids) {
                uu::fill_tensor<int64_t>(
                    m_kvcache_request->get_tensor(m_kvcache_in_ports.at(layer_names::token_type_ids)),
                    0);
            }

            m_generate_initialized = true;
        }

        // NB: KV-cache is full, further generation is impossible
        if (kvcache_desc.num_stored_tokens + input_tokens_len > kvcache_desc.total_size) {
            OPENVINO_THROW("KV-Cache is full.");
        }

        if (const auto deepstack_it = m_kvcache_in_ports.find(layer_names::deepstack_visual_embeds);
            deepstack_it != m_kvcache_in_ports.end()) {
            auto deepstack_local = m_kvcache_request->get_tensor(deepstack_it->second);
            // No visual tokens are generated during the generate stage
            std::fill_n(reinterpret_cast<uint8_t*>(deepstack_local->data()), deepstack_local->get_byte_size(), 0);
        }

        process_longrope(m_kvcache_request, m_kvcache_in_ports, position_ids);
        // FIXME: these tensors should be shared between the parent & child models
        // NB: input_ids can be either fp32(VLM) or i64(LLM)
        auto kv_input_ids = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(m_input_ids_name));
        // NOTE: As `input_tokens_len` can be less than the value of `max_generation_token_len`, which
        //       input layers of generation model are resized to, then we need to put
        //       `input_tokens_len` prompt to the right of `max_generation_token_len`-sized tensors.
        //       Attention mask should rule out all left unusable space.
        std::copy_n(reinterpret_cast<uint8_t*>(input_ids->data()),
                    input_ids->get_byte_size(),
                    reinterpret_cast<uint8_t*>(kv_input_ids->data()) + kv_input_ids->get_byte_size() -
                        input_ids->get_byte_size());

        if (token_type_ids) {
            auto kv_token_type_ids = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(layer_names::token_type_ids));
            util::copy_to_right(token_type_ids, kv_token_type_ids);
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
            std::fill_n(
                kv_attn_mask->data<int64_t>() + kv_attn_mask->get_size() - kvcache_desc.max_generation_token_len,
                kvcache_desc.max_generation_token_len - input_tokens_len,
                0);
        }
        std::fill_n(kv_attn_mask->data<int64_t>() + kv_attn_mask->get_size() - input_tokens_len, input_tokens_len, 1);

        auto kv_pos_ids = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(layer_names::position_ids));
        ov::npuw::util::pad_position_ids(kv_pos_ids, position_ids);

        if (m_eagle3_ext.is_eagle3_model()) {
            m_eagle3_ext.prepare_inputs(m_kvcache_request, m_kvcache_in_ports);
        }

        // Gemma4: pass per_layer_inputs from external inputs into the generate sub-request.
        // Shape is [1, 1, num_layers, proj_dim] during generate; copy right-aligned on seq_len dim.
        if (per_layer_inputs) {
            auto dst = m_kvcache_request->get_tensor(m_kvcache_in_ports.at(layer_names::per_layer_inputs));
            ov::npuw::util::copy_to_right(per_layer_inputs, dst);
        }
    });

    m_llm_profile["N/generate:2.infer"].record([&]() {
        m_kvcache_request->infer();
    });
    kvcache_desc.num_stored_tokens += input_tokens_len;

    // Shared post-infer KV cache update for both lm_head and non-lm_head paths.
    auto do_update_kvcache = [&]() {
        m_llm_profile["N/generate:3.update_kvcache"].record([&]() {
            if (kvcache_desc.num_stored_tokens < kvcache_desc.total_size) {
                m_kvcache_strategy->on_generate_step_done(input_tokens_len);
            }
        });
    };

    if (m_lm_head_request) {
        LOG_DEBUG("Calling inference for LM head model asynchronously");
        m_lm_head_request->start_async();
        do_update_kvcache();
        m_llm_profile["N/generate:4.copy_lincache"].record([&]() {
            copy_lincache(m_kvcache_request, m_kvcache_request, m_kvcache_out_ports, m_kvcache_in_ports);
        });
        m_llm_profile["N/generate:5.lm_head"].record([&]() {
            m_lm_head_request->wait();
            LOG_DEBUG("Calling inference for LM head model -- done.");

            m_logits = m_lm_head_request->get_tensor(m_lm_head_logits_port);
        });
    } else {
        do_update_kvcache();
        m_llm_profile["N/generate:4.copy_lincache"].record([&]() {
            copy_lincache(m_kvcache_request, m_kvcache_request, m_kvcache_out_ports, m_kvcache_in_ports);
        });

        m_logits = m_kvcache_request->get_tensor(m_kvcache_out_ports.at(layer_names::logits));
    }

    if (m_eagle3_ext.is_eagle3_model()) {
        m_eagle3_ext.update_last_hidden_state(m_kvcache_request, m_kvcache_out_ports);
    }

    LOG_DEBUG("Done");
}

void ov::npuw::LLMInferRequest::infer() {
    const auto& inputs = get_inputs();

    auto input_ids = get_tensor(ov::npuw::util::find_port_by_name(inputs, m_input_ids_name).value());
    auto attention_mask = get_tensor(ov::npuw::util::find_port_by_name(inputs, layer_names::attention_mask).value());

    auto position_ids = ov::npuw::util::TensorPtr();
    auto position_ids_opt = ov::npuw::util::find_port_by_name(inputs, layer_names::position_ids);
    if (position_ids_opt.has_value()) {
        position_ids = get_tensor(position_ids_opt.value());
    } else {
        // Sync num_stored_tokens before infer with external updates
        auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
        auto externally_set_num_stored_tokens = m_stored_tokens_state->get_num_stored_tokens();
        if (externally_set_num_stored_tokens > kvcache_desc.total_size) {
            OPENVINO_THROW("Number of updated stored tokens in KV-Cache is greater than total KV-Cache size.");
        }
        if (externally_set_num_stored_tokens < 0) {
            OPENVINO_THROW("Number of updated stored tokens is negative!");
        }
        if (externally_set_num_stored_tokens == 0) {
            m_first_run = true;
        }
        kvcache_desc.num_stored_tokens = static_cast<uint32_t>(externally_set_num_stored_tokens);

        // FIXME: ov::SoPtr<ov::ITensor>?
        position_ids =
            ov::make_tensor(ov::element::i64, ov::Shape{input_ids->get_shape()[0], input_ids->get_shape()[1]});
        std::iota(position_ids->data<int64_t>(),
                  position_ids->data<int64_t>() + position_ids->get_size(),
                  kvcache_desc.num_stored_tokens);
    }

    auto token_type_ids = ov::npuw::util::TensorPtr();
    if (auto type_ids_port = ov::npuw::util::find_port_by_name(inputs, layer_names::token_type_ids);
        type_ids_port.has_value()) {
        token_type_ids = get_tensor(type_ids_port.value());
    }

    auto visual_pos_masks = ov::npuw::util::TensorPtr();
    if (auto port = ov::npuw::util::find_port_by_name(inputs, layer_names::visual_pos_masks); port.has_value()) {
        visual_pos_masks = get_tensor(port.value());
    }

    auto deepstack_visual_embeds = ov::npuw::util::TensorPtr();
    if (auto port = ov::npuw::util::find_port_by_name(inputs, layer_names::deepstack_visual_embeds); port.has_value()) {
        deepstack_visual_embeds = get_tensor(port.value());
    }

    // Gemma4: Extract per_layer_inputs if present
    auto per_layer_inputs = ov::npuw::util::TensorPtr();
    if (auto per_layer_port = ov::npuw::util::find_port_by_name(inputs, layer_names::per_layer_inputs);
        per_layer_port.has_value()) {
        per_layer_inputs = get_tensor(per_layer_port.value());
    }

    // NB: For VLM, the "inputs_embeds" contains float values (embeddings)
    OPENVINO_ASSERT(ov::element::f32 == input_ids->get_element_type() ||
                    ov::element::i64 == input_ids->get_element_type());
    OPENVINO_ASSERT(ov::element::i64 == attention_mask->get_element_type());
    OPENVINO_ASSERT(ov::element::i64 == position_ids->get_element_type());

    // Eagle3: Store Eagle3 specific inputs with pre-validation
    if (m_eagle3_ext.is_eagle3_model()) {
        m_eagle3_ext.store_user_inputs(*this, inputs);
    }

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
    //    both main and draft models for most of LLMs.
    if (input_ids->get_shape()[layer_ids::INPUT_IDS_SEQ_LEN_DIM] > 1 &&
        position_ids->data<int64_t>()[0] == m_first_position_id) {
        infer_prefill(input_ids,
                      attention_mask,
                      position_ids,
                      token_type_ids,
                      per_layer_inputs,
                      visual_pos_masks,
                      deepstack_visual_embeds);
    } else {
        // FIXME: Need to make the solution smarter.
        // Qwen2.5VL uses 3D position_ids but current `trim_kvcache_for_speculative_decoding`
        // doesn't take this into account and causes accuracy issues.
        // Speculative Decode isn't supposed to work with such position_ids currently.
        if (position_ids->get_shape().size() < 3) {
            trim_kvcache_for_speculative_decoding(position_ids);
        }
        infer_generate(input_ids, attention_mask, position_ids, token_type_ids, per_layer_inputs);
    }

    if (!position_ids_opt.has_value()) {
        // Sync num_stored_tokens after infer with internal updates
        m_stored_tokens_state->set_num_stored_tokens(m_npuw_llm_compiled_model->m_kvcache_desc.num_stored_tokens);
    }
}

ov::SoPtr<ov::ITensor> ov::npuw::LLMInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    const auto& port_names = port.get_names();

    if (port_names.count("logits") > 0) {
        if (!m_logits) {
            OPENVINO_THROW("Logits tensor is not available. Please run inference first.");
        }
        return m_logits;
    }

    if (m_eagle3_ext.is_eagle3_model()) {
        if (port_names.count(Eagle3LayerNames::last_hidden_state) > 0) {
            auto last_hidden_state = m_eagle3_ext.get_last_hidden_state();
            if (!last_hidden_state) {
                OPENVINO_THROW("Last hidden state tensor is not available. Please run inference first.");
            }
            return last_hidden_state;
        }
    }

    return ov::ISyncInferRequest::get_tensor(port);
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::LLMInferRequest::query_state() const {
    auto states = m_variableStates;

    if (m_stored_tokens_state) {
        states.push_back(m_stored_tokens_state);
    }

    // Add Eagle3 sampling state if available
    auto eagle3_state = m_eagle3_ext.get_sampling_state();
    if (eagle3_state) {
        states.push_back(eagle3_state);
    }

    return states;
}
