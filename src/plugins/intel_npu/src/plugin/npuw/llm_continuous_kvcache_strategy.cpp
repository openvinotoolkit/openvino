// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_continuous_kvcache_strategy.hpp"

#include "infer_request_utils.hpp"
#include "llm_infer_request.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {

// on_initialize:
//   1. Share past KV buffers from the largest generate variant into all smaller variants
//      so they use the same backing allocation (saves memory).
//   2. Bind prefill past KV to the generate model's buffer so they share memory (chunk prefill).
//   3. Zero-fill the shared KV buffer so the first chunk sees clean state.
// Steps 2–3 are only applicable when chunk prefill is enabled.
void LLMContinuousKVCacheStrategy::on_initialize() {
    // Step 1: share past KV buffers across generate variants.
    // Collect the largest variant's KV tensors into a lookup map (one pass).
    auto& generate_requests = m_req.m_generate_requests;
    if (generate_requests.size() > 1) {
        const auto& largest_request = generate_requests.back();
        std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> largest_past_kv_tensors;
        for (const auto& input_port : largest_request->get_compiled_model()->inputs()) {
            const auto& input_name = input_port.get_any_name();
            if (ov::npuw::util::starts_with(input_name, LLMInferRequest::layer_names::past_key_values)) {
                largest_past_kv_tensors[input_name] = largest_request->get_tensor(input_port);
            }
        }

        // For every smaller variant, share past KV tensors from the largest variant.
        for (size_t i = 0; i < generate_requests.size() - 1; ++i) {
            auto& variant = generate_requests[i];
            for (const auto& input_port : variant->get_compiled_model()->inputs()) {
                const auto& input_name = input_port.get_any_name();
                if (ov::npuw::util::starts_with(input_name, LLMInferRequest::layer_names::past_key_values)) {
                    OPENVINO_ASSERT(largest_past_kv_tensors.find(input_name) != largest_past_kv_tensors.end(),
                                    "Unexpected input name: ",
                                    input_name);
                    auto shared_tensor =
                        ov::SoPtr<ov::ITensor>(ov::make_tensor(input_port.get_element_type(),
                                                               input_port.get_shape(),
                                                               largest_past_kv_tensors.at(input_name)->data()),
                                               nullptr);
                    variant->set_tensor(input_port, shared_tensor);
                }
            }
        }
    }

    // Steps 2–3: bind and zero-fill the shared prefill↔generate KV buffer.
    const bool use_chunk_prefill = m_req.m_npuw_llm_compiled_model->m_use_chunk_prefill;
    if (use_chunk_prefill) {
        m_req.bind_past_kv();
        m_req.clear_chunk_prefill_kv_cache();
    }
}

// on_reset: zero-fill all past KV input tensors in the prefill model.
// next_prompt_length is ignored — continuous strategy has no warm-block concept.
void LLMContinuousKVCacheStrategy::on_reset(uint32_t /*next_prompt_length*/) {
    namespace uu = ov::npuw::util;
    for (const auto& input_name : m_req.m_kvcache_past_names) {
        if (m_req.m_prefill_in_ports.find(input_name) != m_req.m_prefill_in_ports.end()) {
            uu::fill_tensor_bytes(m_req.m_prefill_request->get_tensor(m_req.m_prefill_in_ports.at(input_name)), 0u);
        }
    }
}

// on_prefill_chunk_begin: no-op — continuous KV has no per-chunk setup.
void LLMContinuousKVCacheStrategy::on_prefill_chunk_begin(uint32_t /*current_prompts_len*/) {}

// on_prefill_chunk_done:
//   is_last=false: persist just-inferred KV outputs into past inputs for the next chunk.
//   is_last=true:  leave the KV output in-place in the prefill model's output tensors;
//                  it will be copied into the generate model by on_generate_kv_init()
//                  via copy_kvcache() at the start of the first generate step.
void LLMContinuousKVCacheStrategy::on_prefill_chunk_done(uint32_t current_prompts_len, bool is_last) {
    if (is_last) {
        return;
    }
    const bool v_transposed = m_req.m_npuw_llm_compiled_model->m_kvcache_desc.v_tensors_transposed_pre;
    m_req.update_kvcache_for(m_req.m_prefill_request,
                             m_req.m_prefill_in_ports,
                             m_req.m_prefill_out_ports,
                             current_prompts_len,
                             v_transposed);
}

// on_generate_kv_init: copy the full accumulated prefill KV into the generate
// model's past input buffer so the first generate step sees the correct context.
void LLMContinuousKVCacheStrategy::on_generate_kv_init() {
    m_req.copy_kvcache();
}

// Migrate live KV tokens from the old variant's BNSD layout to the new variant's layout.
// All variants share the same backing buffer but have different per-head strides (S differs),
// so data must be re-packed via a temporary CPU buffer to avoid aliasing corruption.
// Lincache tensors are shared by reference and need no migration.
void LLMContinuousKVCacheStrategy::on_generate_variant_switch(const std::shared_ptr<ov::IAsyncInferRequest>& old_req,
                                                              const PortsMap& old_in_ports,
                                                              const std::shared_ptr<ov::IAsyncInferRequest>& new_req,
                                                              const PortsMap& new_in_ports) {
    namespace uu = ov::npuw::util;
    const auto& kvcache_desc = m_req.m_npuw_llm_compiled_model->m_kvcache_desc;
    const uint32_t num_stored = kvcache_desc.num_stored_tokens;
    if (num_stored == 0) {
        return;
    }

    LOG_DEBUG("Migrating " << num_stored << " KV tokens to new generate variant.");

    for (const auto& name : m_req.m_kvcache_past_names) {
        auto src = old_req->get_tensor(old_in_ports.at(name));
        auto dst = new_req->get_tensor(new_in_ports.at(name));

        // Use the "present" name to distinguish key vs value — same pattern as
        // update_kvcache_for. Direct find("value") on the input name is unreliable
        // because "past_key_values.N.key" contains "value" via "key_values".
        const auto present_name =
            std::regex_replace(name, std::regex(ov::npuw::LLMInferRequest::layer_names::past_key_values), "present");
        const uint32_t kv_dim =
            (present_name.find("value") != std::string::npos && kvcache_desc.v_tensors_transposed_gen)
                ? 3u
                : kvcache_desc.dim;

        auto src_slice = uu::make_tensor_slice(src, kv_dim, 0u, num_stored);
        auto dst_slice = uu::make_tensor_slice(dst, kv_dim, 0u, num_stored);

        // Copy via a temporary CPU buffer to avoid aliasing (src and dst share backing memory).
        auto tmp = uu::allocMem(src->get_element_type(), src_slice->get_shape(), "CPU", nullptr);
        src_slice->copy_to(tmp._ptr);
        uu::copy_tensor_by_dim(tmp, dst_slice, kv_dim, kv_dim);
    }
}

// on_generate_step_done: persist the new token's KV output into the past KV input buffer
// so the next generate step sees the updated context.
void LLMContinuousKVCacheStrategy::on_generate_step_done(uint32_t input_tokens_len) {
    const bool v_transposed = m_req.m_npuw_llm_compiled_model->m_kvcache_desc.v_tensors_transposed_gen;
    m_req.update_kvcache_for(m_req.m_kvcache_request,
                             m_req.m_kvcache_in_ports,
                             m_req.m_kvcache_out_ports,
                             input_tokens_len,
                             v_transposed);
}

}  // namespace npuw
}  // namespace ov
