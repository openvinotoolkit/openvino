// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_continuous_kvcache_strategy.hpp"

#include <regex>

#include "infer_request_utils.hpp"
#include "llm_infer_request.hpp"
#include "llm_prefix_caching.hpp"
#include "openvino/core/parallel.hpp"
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

    // Build the output→input name map for prefix caching once, avoiding repeated
    // regex_replace calls in apply_cached_prefix_blocks().
    // Prefill outputs that contain "present" map to the corresponding prefill input
    // that uses the "past_key_values" prefix instead.
    if (m_req.m_npuw_llm_compiled_model->m_enable_prefix_caching) {
        const std::regex present_regex("present");
        for (const auto& output_port : m_req.m_prefill_request->get_compiled_model()->outputs()) {
            const auto& out_name = output_port.get_any_name();
            if (out_name.find("present") == std::string::npos) {
                continue;
            }
            const auto in_name = std::regex_replace(out_name, present_regex, "past_key_values");
            if (m_req.m_prefill_in_ports.find(in_name) != m_req.m_prefill_in_ports.end()) {
                m_prefill_out_to_in_port_map[out_name] = in_name;
            }
        }
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

// make_prefix_block: copy the prefill output slice [out_token_offset, out_token_offset+block_size) for every
// KV layer into freshly allocated CPU tensors and return them packaged as a KVBlock.
// out_token_offset: token-dimension offset in the padded prefill output buffer
//   (= pad_offset + k*block_size; > 0 when the last chunk is shorter than m_prefill_chunk_size).
// block_token_start: absolute token position in the prompt (used for set_token_start on the block).
std::shared_ptr<KVBlock> LLMContinuousKVCacheStrategy::make_prefix_block(size_t block_token_start,
                                                                         size_t out_token_offset,
                                                                         size_t block_size_arg,
                                                                         const std::vector<uint64_t>& token_hashes) {
    // m_prefill_out_to_in_port_map keys are exactly the KV output names that have a corresponding
    // prefill input port — the same set apply_cached_prefix_blocks will later restore from.
    OPENVINO_ASSERT(!m_prefill_out_to_in_port_map.empty(),
                    "make_prefix_block called but m_prefill_out_to_in_port_map is empty; "
                    "on_initialize() must be called first with prefix caching enabled.");

    const auto& kvcache_desc = m_req.m_npuw_llm_compiled_model->m_kvcache_desc;

    KVData kvcache_data;
    kvcache_data.reserve(m_prefill_out_to_in_port_map.size());

    for (const auto& [output_name, _] : m_prefill_out_to_in_port_map) {
        const bool is_value = (output_name.find("value") != std::string::npos);
        const uint32_t kv_dim = (is_value && kvcache_desc.v_tensors_transposed_pre) ? 3u : kvcache_desc.dim;

        auto kv_src_tensor = m_req.m_prefill_request->get_tensor(m_req.m_prefill_out_ports.at(output_name));
        auto kv_src_slice = ov::npuw::util::view(kv_src_tensor, kv_dim, out_token_offset, block_size_arg);

        auto new_kv_tensor =
            ov::get_tensor_impl(ov::Tensor(kv_src_slice->get_element_type(), kv_src_slice->get_shape()));
        ov::npuw::util::copy_tensor_by_dim(kv_src_slice, new_kv_tensor, kv_dim, kv_dim);

        kvcache_data.emplace_back(output_name, std::move(new_kv_tensor));
    }

    auto block = std::make_shared<KVBlock>(block_size_arg);
    block->set_token_start(block_token_start);
    block->add_block(token_hashes, std::move(kvcache_data));
    return block;
}

// apply_cached_prefix_blocks: copy CPU KV tensors from each cached block's KVData into the
// prefill model's past_key_values input ports.
// The output-to-input name mapping is derived on-the-fly via the same regex used by
// PrefixCachingHelper::create_name_mapping() ("present" → "past_key_values"), with an
// existence check to skip any output that has no corresponding input port.
void LLMContinuousKVCacheStrategy::apply_cached_prefix_blocks(
    const std::vector<std::shared_ptr<KVBlock>>& cached_blocks) {
    if (cached_blocks.empty()) {
        return;
    }

    const auto& kvcache_desc = m_req.m_npuw_llm_compiled_model->m_kvcache_desc;
    const uint64_t block_size = m_req.m_npuw_llm_compiled_model->m_prefix_caching_block_size;

    ov::parallel_for(cached_blocks.size(), [&](size_t block_idx) {
        const auto& block = cached_blocks[block_idx];
        const auto token_start = block->get_token_start();
        const KVData& block_kv_data = block->get_block_kv_data();

        for (const auto& kv_per_layer : block_kv_data) {
            const auto& kv_out_name = kv_per_layer.first;

            auto map_it = m_prefill_out_to_in_port_map.find(kv_out_name);
            if (map_it == m_prefill_out_to_in_port_map.end()) {
                continue;
            }

            auto port_it = m_req.m_prefill_in_ports.find(map_it->second);
            if (port_it == m_req.m_prefill_in_ports.end()) {
                continue;
            }

            const auto& kv_dim =
                (kv_out_name.find("value") != std::string::npos && kvcache_desc.v_tensors_transposed_pre)
                    ? 3u
                    : kvcache_desc.dim;

            auto kv_dst_tensor = m_req.m_prefill_request->get_tensor(port_it->second);
            auto kv_dst_slice = ov::npuw::util::view(kv_dst_tensor, kv_dim, token_start, block_size);
            ov::npuw::util::copy_tensor_by_dim(kv_per_layer.second, kv_dst_slice, kv_dim, kv_dim);
        }
    });
}

}  // namespace npuw
}  // namespace ov
