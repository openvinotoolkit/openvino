// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_swa_cache.hpp"

#include <algorithm>
#include <regex>

#include "infer_request_utils.hpp"
#include "kv_cache_sliding_window_manager.hpp"
#include "llm_infer_request.hpp"
#include "logging.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "util.hpp"

ov::npuw::SwaKVCacheHelper::SwaKVCacheHelper(LLMInferRequest& request, uint32_t window_size)
    : m_request(request),
      m_window_size(window_size) {}

void ov::npuw::SwaKVCacheHelper::share_across_generate_variants() {
    if (m_request.m_swa_past_names.empty()) {
        return;
    }
    auto& generate_requests = m_request.m_generate_requests;
    if (generate_requests.size() <= 1) {
        return;
    }
    const auto& largest_req = generate_requests.back();
    const auto& largest_ports = m_request.m_generate_variant_in_ports.at(largest_req);
    std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> swa_tensors;
    for (const auto& name : m_request.m_swa_past_names) {
        swa_tensors[name] = largest_req->get_tensor(largest_ports.at(name));
    }
    for (size_t i = 0; i < generate_requests.size() - 1; ++i) {
        const auto& variant_ports = m_request.m_generate_variant_in_ports.at(generate_requests[i]);
        for (const auto& name : m_request.m_swa_past_names) {
            const auto& shared_tensor = swa_tensors.at(name);
            auto variant_tensor = generate_requests[i]->get_tensor(variant_ports.at(name));
            OPENVINO_ASSERT(variant_tensor->get_shape() == shared_tensor->get_shape() &&
                                variant_tensor->get_element_type() == shared_tensor->get_element_type(),
                            "SWA past tensor ",
                            name,
                            " shape/type mismatch across generate variants — sliding-window capacity is expected "
                            "to be independent of the total kv-cache size.");
            generate_requests[i]->set_tensor(variant_ports.at(name), shared_tensor);
        }
    }
    LOG_INFO("Shared " << m_request.m_swa_past_names.size() << " SWA tensors across " << generate_requests.size()
                       << " generate variants");
}

void ov::npuw::SwaKVCacheHelper::zero_prefill_tensors() {
    m_request.zero_prefill_past_tensors(m_request.m_swa_past_names);
}

void ov::npuw::SwaKVCacheHelper::copy_prefill_to_generate() {
    if (m_request.m_swa_past_names.empty()) {
        return;
    }
    namespace uu = ov::npuw::util;
    LOG_DEBUG("Copying SWA kv-cache from prefill to generate model.");
    LOG_BLOCK();
    auto& kvcache_desc = m_request.m_npuw_llm_compiled_model->m_kvcache_desc;
    const auto prefill_chunk_size = m_request.m_npuw_llm_compiled_model->m_prefill_chunk_size;
    const bool use_chunk_prefill = m_request.m_npuw_llm_compiled_model->m_use_chunk_prefill;
    const uint32_t last_chunk_tokens =
        use_chunk_prefill ? static_cast<uint32_t>(m_request.m_tokens_in_present_chunk) : 0u;
    OPENVINO_ASSERT(!use_chunk_prefill || kvcache_desc.num_stored_tokens >= last_chunk_tokens,
                    "SWA copy: num_stored_tokens is smaller than last chunk size.");
    const uint32_t tokens_in_past_chunks =
        use_chunk_prefill ? (kvcache_desc.num_stored_tokens - last_chunk_tokens) : 0u;
    const bool chunk_equals_window = use_chunk_prefill && (prefill_chunk_size == m_window_size);

    ov::parallel_for(m_request.m_swa_past_names.size(), [&](size_t out_idx) {
        const auto& input_name = m_request.m_swa_past_names[out_idx];
        auto kvcache_in_tensor = m_request.m_kvcache_request->get_tensor(m_request.m_kvcache_in_ports.at(input_name));

        const auto& output_name =
            std::regex_replace(input_name, std::regex(LLMInferBaseRequest::layer_names::past_key_values), "present");
        OPENVINO_ASSERT(m_request.m_prefill_out_ports.find(output_name) != m_request.m_prefill_out_ports.end(),
                        "Incosistent input/output naming for KV cache: ",
                        output_name,
                        " not found in prefill model outputs.");
        auto prefill_out_tensor =
            m_request.m_prefill_request->get_tensor(m_request.m_prefill_out_ports.at(output_name));

        const auto is_value_tensor = output_name.find("value") != std::string::npos;
        const auto kv_dim = [&](bool v_trans) -> uint32_t {
            return (is_value_tensor && v_trans) ? 3u : kvcache_desc.dim;
        };

        const auto& pre_kv_dim = kv_dim(kvcache_desc.v_tensors_transposed_pre);
        const auto& gen_kv_dim = kv_dim(kvcache_desc.v_tensors_transposed_gen);

        auto append_to_generate_swa = [&](ov::SoPtr<ov::ITensor> src_tensor,
                                          uint32_t src_kv_dim,
                                          uint32_t num_stored_tokens_before,
                                          uint32_t num_new_tokens) {
            if (num_new_tokens == 0u) {
                return;
            }
            uu::write_swa_kv_slice_circular(kvcache_in_tensor,
                                            src_tensor,
                                            gen_kv_dim,
                                            src_kv_dim,
                                            num_stored_tokens_before,
                                            num_new_tokens);
        };

        if (!use_chunk_prefill) {
            // Non-chunk prefill has a single present source; keep the newest num_stored tokens.
            append_to_generate_swa(prefill_out_tensor, pre_kv_dim, 0u, kvcache_desc.num_stored_tokens);
            return;
        }

        // If the last prefill chunk fully spans the SWA window, seeding can be done
        // directly from this chunk's present output.
        if (chunk_equals_window && last_chunk_tokens == prefill_chunk_size) {
            auto prefill_present_kv_chunk =
                uu::make_tensor_slice(prefill_out_tensor, pre_kv_dim, 0u, static_cast<uint32_t>(prefill_chunk_size));
            append_to_generate_swa(prefill_present_kv_chunk, pre_kv_dim, tokens_in_past_chunks, last_chunk_tokens);
            return;
        }

        // Chunk prefill has two logical source segments:
        // 1) persisted past prefix from previous chunks (left-aligned),
        // 2) tail of this loop's present output from the final chunk.

        if (tokens_in_past_chunks > 0u) {
            auto prefill_past_kv = m_request.m_prefill_request->get_tensor(m_request.m_prefill_in_ports.at(input_name));

            // ShrinkSlidingWindowKVCache may shrink SWA past capacity to window size, so
            // read only the valid left-aligned prefix.
            const auto pre_capacity = static_cast<uint32_t>(prefill_past_kv->get_shape()[pre_kv_dim]);
            const uint32_t valid_past_chunks = std::min(tokens_in_past_chunks, pre_capacity);

            if (valid_past_chunks > 0u) {
                ov::SoPtr<ov::ITensor> prefill_past_kv_chunks;
                if (m_request.m_past_kv_bound) {
                    // When prefill/generate past are bound to shared backing, snapshot first
                    // to avoid aliasing while we seed generate SWA.
                    auto tmp_dense_kv_tensor =
                        ov::npuw::util::allocMem(prefill_past_kv->get_element_type(),
                                                 prefill_past_kv->get_shape(),
                                                 m_request.m_pre_alloc_device,
                                                 m_request.m_npuw_llm_compiled_model->get_plugin());
                    NPUW_ASSERT(tmp_dense_kv_tensor._ptr &&
                                "KV cache buffer allocation failed — check device availability and memory constraints");
                    prefill_past_kv->copy_to(tmp_dense_kv_tensor._ptr);
                    prefill_past_kv_chunks =
                        uu::make_tensor_slice(tmp_dense_kv_tensor, pre_kv_dim, 0u, valid_past_chunks);
                } else {
                    prefill_past_kv_chunks = uu::make_tensor_slice(prefill_past_kv, pre_kv_dim, 0u, valid_past_chunks);
                }
                append_to_generate_swa(prefill_past_kv_chunks, pre_kv_dim, 0u, tokens_in_past_chunks);
            }
        }

        if (last_chunk_tokens > 0u) {
            auto prefill_present_kv_chunk =
                uu::make_tensor_slice(prefill_out_tensor,
                                      pre_kv_dim,
                                      static_cast<uint32_t>(prefill_chunk_size - last_chunk_tokens),
                                      static_cast<uint32_t>(prefill_chunk_size));
            append_to_generate_swa(prefill_present_kv_chunk, pre_kv_dim, tokens_in_past_chunks, last_chunk_tokens);
        }
    });
    LOG_DEBUG("Done.");
}

void ov::npuw::SwaKVCacheHelper::update_cache(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                              const PortsMap& in_ports,
                                              const PortsMap& out_ports,
                                              uint32_t num_tokens,
                                              bool v_transposed,
                                              bool use_circular_layout) {
    if (m_request.m_swa_past_names.empty()) {
        return;
    }
    namespace uu = ov::npuw::util;
    auto& kvcache_desc = m_request.m_npuw_llm_compiled_model->m_kvcache_desc;

    ov::parallel_for(m_request.m_swa_past_names.size(), [&](size_t i) {
        const auto& input_name = m_request.m_swa_past_names[i];
        OPENVINO_ASSERT(in_ports.find(input_name) != in_ports.end(),
                        "There is no ",
                        input_name,
                        " in input ports map, while it is expected!");
        const auto& output_name =
            std::regex_replace(input_name, std::regex(LLMInferBaseRequest::layer_names::past_key_values), "present");
        OPENVINO_ASSERT(out_ports.find(output_name) != out_ports.end(),
                        "There is no ",
                        output_name,
                        " in output ports map, while it is expected!");

        auto dst_tensor = request->get_tensor(in_ports.at(input_name));
        const auto& kv_dim = (output_name.find("value") != std::string::npos && v_transposed) ? 3u : kvcache_desc.dim;
        auto src_tensor = request->get_tensor(out_ports.at(output_name));

        // num_stored_tokens was already advanced by the caller past the tokens written here.
        const uint32_t num_stored_before = kvcache_desc.num_stored_tokens - num_tokens;
        if (use_circular_layout) {
            uu::write_swa_kv_slice_circular(dst_tensor, src_tensor, kv_dim, kv_dim, num_stored_before, num_tokens);
        } else {
            uu::write_swa_kv_slice_left_aligned(dst_tensor, src_tensor, kv_dim, kv_dim, num_stored_before, num_tokens);
        }
    });
}

void ov::npuw::SwaKVCacheHelper::update_prefill(uint32_t num_tokens) {
    auto& kvcache_desc = m_request.m_npuw_llm_compiled_model->m_kvcache_desc;
    update_cache(m_request.m_prefill_request,
                 m_request.m_prefill_in_ports,
                 m_request.m_prefill_out_ports,
                 num_tokens,
                 kvcache_desc.v_tensors_transposed_pre,
                 /*use_circular_layout=*/false);
}

void ov::npuw::SwaKVCacheHelper::update_generate(uint32_t num_tokens) {
    auto& kvcache_desc = m_request.m_npuw_llm_compiled_model->m_kvcache_desc;
    update_cache(m_request.m_kvcache_request,
                 m_request.m_kvcache_in_ports,
                 m_request.m_kvcache_out_ports,
                 num_tokens,
                 kvcache_desc.v_tensors_transposed_gen,
                 /*use_circular_layout=*/true);
}

void ov::npuw::SwaKVCacheHelper::fill_attention_masks(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                                      const PortsMap& in_ports,
                                                      uint32_t num_new_tokens) const {
    auto& kvcache_desc = m_request.m_npuw_llm_compiled_model->m_kvcache_desc;
    const uint32_t num_stored_before = kvcache_desc.num_stored_tokens;
    ov::npuw::util::fill_sliding_window_attention_mask(request,
                                                       in_ports,
                                                       num_stored_before,
                                                       num_new_tokens,
                                                       m_window_size);
}
