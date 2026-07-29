// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_continuous_kvcache_strategy.hpp"

#include <atomic>
#include <chrono>
#include <regex>
#include <unordered_map>

#include "infer_request_utils.hpp"
#include "llm_infer_request.hpp"
#include "logging.hpp"
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

namespace {

// Plan for continuing a prefill on the contiguous buffer. The live KV prefix can sit
// in two places. If generation ran, the generate model's past inputs hold the whole
// history in the generate layout and must be repacked into the prefill layout. If the
// request finished straight from the prompt logits, the decode loop never executed,
// so the prefix is split between the prefill past inputs and the last chunk still
// sitting in the prefill present outputs.
struct ContinuousContinuationPlan final : ContinuedPrefillPlan {
    uint32_t keep = 0u;
    bool source_is_generate = false;
    // Used when the source is the prefill request itself.
    uint32_t past_tokens = 0u;
    uint32_t present_tokens = 0u;
    // Staging buffers for the alias-safe repack, one per KV input that needs it.
    std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> temps;
};

}  // anonymous namespace

std::unique_ptr<ContinuedPrefillPlan> LLMContinuousKVCacheStrategy::plan_continued_prefill(uint32_t keep,
                                                                                           uint32_t delta_len) {
    namespace uu = ov::npuw::util;
    const auto& compiled = m_req.m_npuw_llm_compiled_model;
    const auto& kvcache_desc = compiled->m_kvcache_desc;

    OPENVINO_ASSERT(compiled->m_use_chunk_prefill, "Continued prefill requires chunked prefill.");
    OPENVINO_ASSERT(keep > 0u && keep + delta_len <= kvcache_desc.max_prompt_size,
                    "Continued prefill: keep (",
                    keep,
                    ") plus delta (",
                    delta_len,
                    ") must fit into the maximum prompt size (",
                    kvcache_desc.max_prompt_size,
                    ").");
    OPENVINO_ASSERT(keep <= kvcache_desc.max_prompt_size - compiled->m_prefill_chunk_size,
                    "Continued prefill: keep exceeds the prefill past KV view capacity.");

    auto plan = std::make_unique<ContinuousContinuationPlan>();
    plan->keep = keep;
    plan->source_is_generate = m_req.m_generate_initialized;

    if (plan->source_is_generate) {
        // Validate every layer before touching any of them and pre-allocate the staging
        // buffers where the generate and prefill views share backing memory.
        for (const auto& name : m_req.m_kvcache_past_names) {
            OPENVINO_ASSERT(m_req.m_kvcache_in_ports.count(name) && m_req.m_prefill_in_ports.count(name),
                            "Continued prefill: KV input ",
                            name,
                            " is missing from the generate or prefill port map.");
            auto src = m_req.m_kvcache_request->get_tensor(m_req.m_kvcache_in_ports.at(name));
            auto dst = m_req.m_prefill_request->get_tensor(m_req.m_prefill_in_ports.at(name));
            OPENVINO_ASSERT(src->get_element_type() == dst->get_element_type(),
                            "Continued prefill: element type mismatch for ",
                            name);

            const auto present_name =
                std::regex_replace(name, std::regex(LLMInferRequest::layer_names::past_key_values), "present");
            const bool is_value = present_name.find("value") != std::string::npos;
            const uint32_t src_dim = (is_value && kvcache_desc.v_tensors_transposed_gen) ? 3u : kvcache_desc.dim;
            const uint32_t dst_dim = (is_value && kvcache_desc.v_tensors_transposed_pre) ? 3u : kvcache_desc.dim;
            OPENVINO_ASSERT(src->get_shape()[src_dim] >= keep && dst->get_shape()[dst_dim] >= keep,
                            "Continued prefill: KV extent of ",
                            name,
                            " cannot cover the granted keep.");

            // Shared backing memory is an aliasing hazard, not layout compatibility.
            // Identical bytes are addressed with different per-head strides whenever the
            // sequence extents differ, so the repack must stage through a temporary.
            const bool aliased = src->data() == dst->data();
            const bool same_layout = aliased && src->get_shape() == dst->get_shape() && src_dim == dst_dim;
            if (aliased && !same_layout) {
                auto src_slice = uu::make_tensor_slice(src, src_dim, 0u, keep);
                plan->temps[name] = uu::allocMem(src->get_element_type(), src_slice->get_shape(), "CPU", nullptr);
            }
            if (name == m_req.m_kvcache_past_names.front()) {
                // One line of layout diagnosis per continuation so slow copy paths can be
                // identified from the log alone.
                LOG_INFO("Continued prefill layout: src "
                         << src->get_shape() << " dim " << src_dim << ", dst " << dst->get_shape() << " dim " << dst_dim
                         << ", aliased " << aliased << ", same_layout " << same_layout << ", v_trans gen/pre "
                         << kvcache_desc.v_tensors_transposed_gen << "/" << kvcache_desc.v_tensors_transposed_pre);
            }
        }
    } else {
        // The prompt-phase finish case. The prefill past inputs hold everything except
        // the last chunk, which is still in the present outputs.
        const uint32_t live = kvcache_desc.num_stored_tokens;
        plan->present_tokens = static_cast<uint32_t>(m_req.m_tokens_in_present_chunk);
        OPENVINO_ASSERT(live >= plan->present_tokens, "Continued prefill: inconsistent stored token accounting.");
        plan->past_tokens = live - plan->present_tokens;
        OPENVINO_ASSERT(keep <= live, "Continued prefill: keep exceeds the live token count.");
        for (const auto& name : m_req.m_kvcache_past_names) {
            const auto present_name =
                std::regex_replace(name, std::regex(LLMInferRequest::layer_names::past_key_values), "present");
            OPENVINO_ASSERT(m_req.m_prefill_in_ports.count(name) && m_req.m_prefill_out_ports.count(present_name),
                            "Continued prefill: prefill ports are missing for ",
                            name);
        }
    }
    return plan;
}

void LLMContinuousKVCacheStrategy::apply_continued_prefill(ContinuedPrefillPlan& base_plan) {
    namespace uu = ov::npuw::util;
    auto& plan = static_cast<ContinuousContinuationPlan&>(base_plan);
    const auto& kvcache_desc = m_req.m_npuw_llm_compiled_model->m_kvcache_desc;
    const uint32_t keep = plan.keep;

    if (plan.source_is_generate) {
        LOG_DEBUG("Continued prefill: repacking " << keep << " tokens from generate past KV into prefill layout.");
        const auto t_start = std::chrono::steady_clock::now();
        std::atomic<uint64_t> bytes_total{0u};
        std::atomic<uint32_t> n_skipped{0u}, n_staged{0u}, n_permuted{0u}, n_direct{0u};
        // Every tensor is independent, so repack them in parallel like copy_kvcache does.
        // The plan's temp map is only read here.
        ov::parallel_for(m_req.m_kvcache_past_names.size(), [&](size_t idx) {
            const auto& name = m_req.m_kvcache_past_names[idx];
            auto src = m_req.m_kvcache_request->get_tensor(m_req.m_kvcache_in_ports.at(name));
            auto dst = m_req.m_prefill_request->get_tensor(m_req.m_prefill_in_ports.at(name));

            const auto present_name =
                std::regex_replace(name, std::regex(LLMInferRequest::layer_names::past_key_values), "present");
            const bool is_value = present_name.find("value") != std::string::npos;
            const uint32_t src_dim = (is_value && kvcache_desc.v_tensors_transposed_gen) ? 3u : kvcache_desc.dim;
            const uint32_t dst_dim = (is_value && kvcache_desc.v_tensors_transposed_pre) ? 3u : kvcache_desc.dim;

            const bool aliased = src->data() == dst->data();
            const bool same_layout = aliased && src->get_shape() == dst->get_shape() && src_dim == dst_dim;
            if (same_layout) {
                // The bytes already sit where the prefill will read them.
                n_skipped.fetch_add(1u, std::memory_order_relaxed);
                return;
            }

            auto src_slice = uu::make_tensor_slice(src, src_dim, 0u, keep);
            auto dst_slice = uu::make_tensor_slice(dst, dst_dim, 0u, keep);
            bytes_total.fetch_add(src_slice->get_byte_size(), std::memory_order_relaxed);
            if (src_dim != dst_dim) {
                n_permuted.fetch_add(1u, std::memory_order_relaxed);
            } else {
                n_direct.fetch_add(1u, std::memory_order_relaxed);
            }
            auto temp_it = plan.temps.find(name);
            if (temp_it != plan.temps.end()) {
                n_staged.fetch_add(1u, std::memory_order_relaxed);
                src_slice->copy_to(temp_it->second._ptr);
                uu::copy_tensor_by_dim(temp_it->second, dst_slice, src_dim, dst_dim);
            } else {
                uu::copy_tensor_by_dim(src_slice, dst_slice, src_dim, dst_dim);
            }
        });
        const auto t_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_start).count() /
            1000.0;
        LOG_INFO("Continued prefill repack: " << keep << " tokens, " << (bytes_total.load() / 1024) << " KiB in "
                                              << t_ms << " ms (tensors: " << n_direct.load() << " direct, "
                                              << n_permuted.load() << " permuted, " << n_staged.load() << " staged, "
                                              << n_skipped.load() << " skipped)");
    } else if (keep > plan.past_tokens) {
        // Persist the final present chunk into the past inputs before the next prefill
        // overwrites it. Repacking from the generate request here would copy stale data
        // over a valid prefix.
        LOG_DEBUG("Continued prefill: persisting " << plan.present_tokens
                                                   << " present-chunk tokens into prefill past KV.");
        const auto chunk = static_cast<uint32_t>(m_req.m_npuw_llm_compiled_model->m_prefill_chunk_size);
        ov::parallel_for(m_req.m_kvcache_past_names.size(), [&](size_t idx) {
            const auto& name = m_req.m_kvcache_past_names[idx];
            const auto present_name =
                std::regex_replace(name, std::regex(LLMInferRequest::layer_names::past_key_values), "present");
            const bool is_value = present_name.find("value") != std::string::npos;
            const uint32_t kv_dim = (is_value && kvcache_desc.v_tensors_transposed_pre) ? 3u : kvcache_desc.dim;

            auto present = m_req.m_prefill_request->get_tensor(m_req.m_prefill_out_ports.at(present_name));
            auto past = m_req.m_prefill_request->get_tensor(m_req.m_prefill_in_ports.at(name));

            auto src_slice = uu::make_tensor_slice(present, kv_dim, chunk - plan.present_tokens, chunk);
            auto dst_slice =
                uu::make_tensor_slice(past, kv_dim, plan.past_tokens, plan.past_tokens + plan.present_tokens);
            uu::copy_tensor_by_dim(src_slice, dst_slice, kv_dim, kv_dim);
        });
    }
    // Otherwise the prefill past inputs already hold the whole preserved prefix.
}

}  // namespace npuw
}  // namespace ov
