// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention.hpp"

#include <array>
#include <cctype>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <vector>
#include <memory>
#include <algorithm>
#include <utility>

#include "common_utils/jitter.hpp"
#include "common_utils/kernel_generator_base.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "kv_cache_inst.h"
#include "openvino/core/partial_shape.hpp"
#include "paged_attention_gen.hpp"
#include "paged_attention_inst.h"
#include "primitive_cm_base.hpp"
#include "primitive_inst.h"

#define DUMP_XATTN_INTERNALS 0
#if DUMP_XATTN_INTERNALS
#    include "openvino/util/file_util.hpp"
#    define XATTN_DUMP(instance, stage) dump_xattn_internals((instance), (stage))
#else
#    define XATTN_DUMP(instance, stage) ((void)0)
#endif

namespace ov::intel_gpu::cm {

namespace {

std::vector<int32_t> read_subsequence_begins(const kernel_impl_params& params) {
    const auto& memory_deps = params.memory_deps;
    const auto subsequence_begins_mem = memory_deps.at(PagedAttentionInputIdx::SUBSEQUENCE_BEGINS);
    mem_lock<int32_t, mem_lock_type::read> lock(subsequence_begins_mem, *params.strm);
    return std::vector<int32_t>(lock.begin(), lock.end());
}

std::vector<int32_t> read_past_lens(const kernel_impl_params& params) {
    const auto& memory_deps = params.memory_deps;
    const auto past_lens_mem = memory_deps.at(PagedAttentionInputIdx::PAST_LENS);
    mem_lock<int32_t, mem_lock_type::read> lock(past_lens_mem, *params.strm);
    return std::vector<int32_t>(lock.begin(), lock.end());
}

MixedRouteMode get_mixed_route_mode_from_env() {
    const char* env = std::getenv("OV_GPU_CM_MIXED_ROUTE_MODE");
    if (!env) {
        return MixedRouteMode::MULTI;
    }

    std::string mode(env);
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (mode == "split" || mode == "route_split") {
        return MixedRouteMode::SPLIT;
    }

    return MixedRouteMode::MULTI;
}

size_t get_batch_size_in_sequences(const kernel_impl_params& params) {
    const auto& memory_deps = params.memory_deps;
    const auto past_lens_mem = memory_deps.at(PagedAttentionInputIdx::PAST_LENS);
    mem_lock<int32_t, mem_lock_type::read> lock(past_lens_mem, *params.strm);
    return lock.size();
}

bool has_multiple_subsequences(const kernel_impl_params& params) {
    return get_batch_size_in_sequences(params) > 1;
}

bool is_cm_pa_exec_probe_enabled() {
    const char* env = std::getenv("OV_GPU_CM_PA_EXEC_PROBE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool is_cm_pa_kv_detail_probe_enabled() {
    const char* env = std::getenv("OV_GPU_CM_PA_EXEC_PROBE_KV_DETAIL");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool is_cm_pa_force_lockable_mapping_enabled() {
    const char* env = std::getenv("OV_GPU_CM_PA_FORCE_LOCKABLE_MAPPING");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

}  // namespace

class PagedAttentionCmImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::PagedAttentionCmImpl)

    Stage::Ptr kv_cache_update = make_stage<PagedAttentionGeneratorKVCacheUpdate>();
    Stage::Ptr pa_single_token = make_stage<PagedAttentionGeneratorSingleToken>();
    Stage::Ptr pa_single_token_finalization = make_stage<PagedAttentionGeneratorSingleTokenFinalization>();
    Stage::Ptr pa_multi_token_1 = make_stage<PagedAttentionGeneratorMultiToken>(1);
    Stage::Ptr pa_multi_token_128 = make_stage<PagedAttentionGeneratorMultiToken>(128);
    Stage::Ptr pa_multi_token_256 = make_stage<PagedAttentionGeneratorMultiToken>(256);
    Stage::Ptr xattn_estimate_gemmqk = make_stage<XAttentionEstimateGEMMQK>(128);
    Stage::Ptr xattn_estimate_find_block = make_stage<XAttentionEstimateFindBlock>(128);
    Stage::Ptr xattn_estimate_post_proc = make_stage<XAttentionEstimatePostProc>(128);
    Stage::Ptr xattn_estimate_gemmqk_256 = make_stage<XAttentionEstimateGEMMQK>(256);
    Stage::Ptr xattn_estimate_find_block_256 = make_stage<XAttentionEstimateFindBlock>(256);
    Stage::Ptr xattn_estimate_post_proc_256 = make_stage<XAttentionEstimatePostProc>(256);

    PagedAttentionCmImpl() : PrimitiveImplCM(PagedAttentionImplementationManager::get_type_info_static()) {
        m_rt_params = std::make_unique<PagedAttentionRuntimeParams>();
    }
    explicit PagedAttentionCmImpl(const kernel_impl_params& params) : PagedAttentionCmImpl() {
        const auto desc = params.typed_desc<paged_attention>();

        GPU_DEBUG_TRACE_DETAIL << "ov::intel_gpu::cm::PagedAttentionCmImpl::PagedAttentionCmImpl()" << std::endl;
        add_stage(kv_cache_update, params);
        add_stage(pa_single_token, params);
        add_stage(pa_single_token_finalization, params);
        add_stage(pa_multi_token_1, params);
        if (desc->has_xattention) {
            add_stage(pa_multi_token_128, params);
            if (params.get_device_info().arch >= gpu_arch::xe2) {
                add_stage(pa_multi_token_256, params);
            }

            add_stage(xattn_estimate_gemmqk, params);
            add_stage(xattn_estimate_find_block, params);
            add_stage(xattn_estimate_post_proc, params);

            if (params.get_device_info().arch >= gpu_arch::xe2) {
                add_stage(xattn_estimate_gemmqk_256, params);
                add_stage(xattn_estimate_find_block_256, params);
                add_stage(xattn_estimate_post_proc_256, params);
            }
        }
    }

    void update_xattn_rt_params(const kernel_impl_params& params) {
        const auto desc = params.typed_desc<paged_attention>();

        // XAttention estimate is following afer kvcache_update.
        auto out_shape = params.output_layouts[0].get_shape();
        const size_t block_size = get_xattn_block_size(params);
        const uint32_t block_wg_n = XAttentionEstimateGeneratorBase::get_block_wg_n(params);
        const uint32_t block_wg_m = XAttentionEstimateGeneratorBase::get_block_wg_m(params);
        const size_t kv_len = get_max_context_len(params);
        const size_t q_len = out_shape[0];
        const size_t N = kv_len / STRIDE;
        const size_t N_kq_groups = ceil_div(N, block_wg_n);

        const auto q_block_pad = ceil_div(q_len, block_size);
        const auto sum_per_token_in_block = block_size / STRIDE;
        const auto k_block_in_group = block_wg_n / sum_per_token_in_block;
        const auto k_block_pad = k_block_in_group * N_kq_groups;

        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        rt_params->block_wg_m = block_wg_m;
        rt_params->q_block_pad = q_block_pad;
        rt_params->k_block_pad = k_block_pad;

        const size_t merged_q_num = PagedAttentionGeneratorMultiToken::get_wg_seq_len(params) / block_size;
        rt_params->q_block_pad_merged = ceil_div(q_block_pad, merged_q_num);

        const size_t head_size = desc->k_head_size;

        const auto M = q_len / STRIDE;  //# will slient drop the tails which is less than `stride`
        const auto K = STRIDE * head_size;

        const size_t q_stride_pad = round_up_to(M, block_wg_m);

        rt_params->N_kq_groups = N_kq_groups;
        rt_params->M = M;
        rt_params->N = N;
        rt_params->K = K;
        rt_params->q_stride_pad = q_stride_pad;
        rt_params->xattn_block_size = block_size;
    }

    void update_rt_params(const primitive_inst& instance) override {
        update_stages_flags(instance);
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<PagedAttentionRuntimeParams>();
        }

        const auto& params = *instance.get_impl_params();
        OPENVINO_ASSERT(!params.is_dynamic());
        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        const auto& desc = params.typed_desc<paged_attention>();
        rt_params->batch_size_in_sequences = get_batch_size_in_sequences(params);
        rt_params->single_token_selected_count = 0;
        rt_params->multi_token_wg_count = 0;
        rt_params->enable_xattn_estimation = false;

        rt_params->stage = get_paged_attention_stage(params);
        rt_params->mixed_route_mode = get_mixed_route_mode_from_env();
        const auto max_context_len = get_max_context_len(params);
        rt_params->max_context_len = max_context_len;
        GPU_DEBUG_TRACE_DETAIL << "update_rt_params for stage: " << static_cast<size_t>(rt_params->stage) << "  max_context_len: " << rt_params->max_context_len
                               << std::endl;

        if (rt_params->stage == PagedAttentionStage::GENERATE) {
            auto partition_size = PagedAttentionGeneratorSingleToken::get_partition_size(desc->has_xattention);
            rt_params->num_of_partitions = ceil_div(max_context_len, partition_size);
            rt_params->q_chunking = get_single_token_q_chunking(params, *desc, partition_size);
            rt_params->single_token_selected_count = rt_params->batch_size_in_sequences;
            GPU_DEBUG_TRACE_DETAIL << "  partition_size: " << partition_size << "  num_of_partitions: " << rt_params->num_of_partitions << std::endl;
        } else {
            const auto subsequence_begins = read_subsequence_begins(params);
            const auto past_lens = read_past_lens(params);
            const auto wg_seq_len = PagedAttentionGeneratorMultiToken::get_wg_seq_len(params);
            const bool use_split_mixed = rt_params->stage == PagedAttentionStage::MIXED &&
                                         rt_params->mixed_route_mode == MixedRouteMode::SPLIT;
            for (size_t subsequence_id = 0; subsequence_id + 1 < subsequence_begins.size(); ++subsequence_id) {
                const auto q_len = static_cast<size_t>(std::max<int32_t>(subsequence_begins[subsequence_id + 1] - subsequence_begins[subsequence_id], 0));
                if (q_len == 0) {
                    continue;
                }

                if (use_split_mixed) {
                    const auto past_len = static_cast<size_t>(std::max<int32_t>(past_lens[subsequence_id], 0));
                    const bool decode_subsequence = (q_len == 1) && (past_len > 0);
                    if (decode_subsequence) {
                        rt_params->single_token_selected_count++;
                    } else {
                        rt_params->multi_token_wg_count += ceil_div(q_len, wg_seq_len);
                    }
                } else {
                    rt_params->multi_token_wg_count += ceil_div(q_len, wg_seq_len);
                }
            }

            if (use_split_mixed && rt_params->single_token_selected_count > 0) {
                const auto partition_size = PagedAttentionGeneratorSingleToken::get_partition_size(desc->has_xattention);
                rt_params->num_of_partitions = ceil_div(max_context_len, partition_size);
                rt_params->q_chunking = get_single_token_q_chunking(params, *desc, partition_size);
            }

            if (desc->has_xattention) {
                validate_xattn_inputs(params, rt_params->batch_size_in_sequences);
            }

            if (desc->has_xattention && rt_params->batch_size_in_sequences == 1) {
                rt_params->enable_xattn_estimation = true;
                update_xattn_rt_params(params);
            } else {
                rt_params->xattn_block_size = 1;  // disable xattn for pa
            }

            if (use_split_mixed) {
                // In split route mode, mixed stage follows conservative non-xattention behavior.
                rt_params->enable_xattn_estimation = false;
                rt_params->xattn_block_size = 1;
            }
        }
    }

    void prepare_multi_token_mapping(primitive_inst& instance) {
        const auto& params = *instance.get_impl_params();
        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        OPENVINO_ASSERT(rt_params != nullptr);
        if (rt_params->multi_token_wg_count == 0) {
            return;
        }
        auto& stream = instance.get_network().get_stream();

        const bool use_split_mixed = rt_params->stage == PagedAttentionStage::MIXED &&
                                     rt_params->mixed_route_mode == MixedRouteMode::SPLIT;

        const auto subsequence_begins = read_subsequence_begins(params);
        std::vector<int32_t> past_lens;
        if (use_split_mixed) {
            past_lens = read_past_lens(params);
        }

        const auto wg_seq_len = static_cast<int32_t>(PagedAttentionGeneratorMultiToken::get_wg_seq_len(params));
        auto mapping_mem = instance.get_intermediates_memories()[PagedAttentionInternBuffIdx::MULTI_TOKEN_WG_MAPPING];
        const bool force_lockable = is_cm_pa_force_lockable_mapping_enabled();
        const auto alloc_type = mapping_mem->get_allocation_type();
        const bool use_copy_from = !force_lockable && alloc_type == cldnn::allocation_type::usm_device;

        if (force_lockable) {
            OPENVINO_ASSERT(alloc_type != cldnn::allocation_type::usm_device,
                            "prepare_multi_token_mapping requires lockable memory when OV_GPU_CM_PA_FORCE_LOCKABLE_MAPPING is enabled");
            mem_lock<int32_t, mem_lock_type::write> mapping_lock(mapping_mem, stream);
            size_t mapping_idx = 0;
            for (size_t subsequence_id = 0; subsequence_id + 1 < subsequence_begins.size(); ++subsequence_id) {
                const int32_t q_begin = subsequence_begins[subsequence_id];
                const int32_t q_end = subsequence_begins[subsequence_id + 1];

                if (q_end <= q_begin) {
                    continue;
                }

                if (use_split_mixed) {
                    const int32_t q_len = q_end - q_begin;
                    const int32_t past_len = std::max<int32_t>(past_lens[subsequence_id], 0);
                    const bool decode_subsequence = (q_len == 1) && (past_len > 0);
                    if (decode_subsequence) {
                        continue;
                    }
                }

                for (int32_t block_start = q_begin; block_start < q_end; block_start += wg_seq_len) {
                    mapping_lock[mapping_idx++] = block_start;
                    mapping_lock[mapping_idx++] = static_cast<int32_t>(subsequence_id);
                }
            }

            OPENVINO_ASSERT(mapping_idx == rt_params->multi_token_wg_count * 2,
                            "Unexpected multi-token mapping size: expected ",
                            rt_params->multi_token_wg_count * 2,
                            ", got ",
                            mapping_idx);
        } else {
            std::vector<int32_t> mapping(rt_params->multi_token_wg_count * 2, 0);
            size_t mapping_idx = 0;

            for (size_t subsequence_id = 0; subsequence_id + 1 < subsequence_begins.size(); ++subsequence_id) {
                const int32_t q_begin = subsequence_begins[subsequence_id];
                const int32_t q_end = subsequence_begins[subsequence_id + 1];

                if (q_end <= q_begin) {
                    continue;
                }

                if (use_split_mixed) {
                    const int32_t q_len = q_end - q_begin;
                    const int32_t past_len = std::max<int32_t>(past_lens[subsequence_id], 0);
                    const bool decode_subsequence = (q_len == 1) && (past_len > 0);
                    if (decode_subsequence) {
                        continue;
                    }
                }

                for (int32_t block_start = q_begin; block_start < q_end; block_start += wg_seq_len) {
                    mapping[mapping_idx++] = block_start;
                    mapping[mapping_idx++] = static_cast<int32_t>(subsequence_id);
                }
            }

            OPENVINO_ASSERT(mapping_idx == rt_params->multi_token_wg_count * 2,
                            "Unexpected multi-token mapping size: expected ",
                            rt_params->multi_token_wg_count * 2,
                            ", got ",
                            mapping_idx);

            if (!use_copy_from) {
                mem_lock<int32_t, mem_lock_type::write> mapping_lock(mapping_mem, stream);
                std::copy(mapping.begin(), mapping.end(), mapping_lock.begin());
            } else {
                mapping_mem->copy_from(stream, mapping.data(), 0, 0, mapping.size() * sizeof(int32_t), true);
            }
        }
    }

    void prepare_single_token_selected_ids(primitive_inst& instance) {
        const auto& params = *instance.get_impl_params();
        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        OPENVINO_ASSERT(rt_params != nullptr);
        auto& stream = instance.get_network().get_stream();

        auto selected_ids_mem = instance.get_intermediates_memories()[PagedAttentionInternBuffIdx::SINGLE_TOKEN_SELECTED_SEQ_IDS];
        const bool force_lockable = is_cm_pa_force_lockable_mapping_enabled();
        const auto alloc_type = selected_ids_mem->get_allocation_type();
        const bool use_copy_from = !force_lockable && alloc_type == cldnn::allocation_type::usm_device;
        size_t selected_count = 0;

        const bool use_split_mixed = rt_params->stage == PagedAttentionStage::MIXED &&
                                     rt_params->mixed_route_mode == MixedRouteMode::SPLIT;
        if (force_lockable) {
            OPENVINO_ASSERT(alloc_type != cldnn::allocation_type::usm_device,
                            "prepare_single_token_selected_ids requires lockable memory when OV_GPU_CM_PA_FORCE_LOCKABLE_MAPPING is enabled");
            mem_lock<int32_t, mem_lock_type::write> selected_ids_lock(selected_ids_mem, stream);

            if (use_split_mixed) {
                const auto subsequence_begins = read_subsequence_begins(params);
                const auto past_lens = read_past_lens(params);
                for (size_t sequence_id = 0; sequence_id + 1 < subsequence_begins.size(); ++sequence_id) {
                    const int32_t q_len = subsequence_begins[sequence_id + 1] - subsequence_begins[sequence_id];
                    const int32_t past_len = std::max<int32_t>(past_lens[sequence_id], 0);
                    if (q_len == 1 && past_len > 0) {
                        selected_ids_lock[selected_count++] = static_cast<int32_t>(sequence_id);
                    }
                }
            } else {
                for (size_t sequence_id = 0; sequence_id < rt_params->batch_size_in_sequences; ++sequence_id) {
                    selected_ids_lock[selected_count++] = static_cast<int32_t>(sequence_id);
                }
            }

        } else {
            std::vector<int32_t> selected_ids(rt_params->batch_size_in_sequences, 0);
            if (use_split_mixed) {
                const auto subsequence_begins = read_subsequence_begins(params);
                const auto past_lens = read_past_lens(params);
                for (size_t sequence_id = 0; sequence_id + 1 < subsequence_begins.size(); ++sequence_id) {
                    const int32_t q_len = subsequence_begins[sequence_id + 1] - subsequence_begins[sequence_id];
                    const int32_t past_len = std::max<int32_t>(past_lens[sequence_id], 0);
                    if (q_len == 1 && past_len > 0) {
                        selected_ids[selected_count++] = static_cast<int32_t>(sequence_id);
                    }
                }
            } else {
                for (size_t sequence_id = 0; sequence_id < rt_params->batch_size_in_sequences; ++sequence_id) {
                    selected_ids[selected_count++] = static_cast<int32_t>(sequence_id);
                }
            }

            if (selected_count > 0) {
                if (!use_copy_from) {
                    mem_lock<int32_t, mem_lock_type::write> selected_ids_lock(selected_ids_mem, stream);
                    std::copy(selected_ids.begin(), selected_ids.begin() + selected_count, selected_ids_lock.begin());
                } else {
                    selected_ids_mem->copy_from(stream, selected_ids.data(), 0, 0, selected_count * sizeof(int32_t), true);
                }
            }
        }

        if (is_cm_pa_exec_probe_enabled()) {
            std::fprintf(stderr,
                         "[CM_PA_PREP_PROBE] fn=prepare_single_token_selected_ids force_lockable=%d alloc_type=%d size=%zu method=%s\n",
                         force_lockable ? 1 : 0,
                         static_cast<int>(alloc_type),
                         selected_count,
                         use_copy_from ? "copy_from" : "mem_lock_write");
        }

        rt_params->single_token_selected_count = selected_count;
    }

    // update impl_parameter and rt_parameter
    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        PrimitiveImplCM::update(inst, impl_params);
        update_rt_params(inst);
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        const auto& params = *instance.get_impl_params();
        const auto desc = params.typed_desc<paged_attention>();
        using probe_clock = std::chrono::steady_clock;

        update_stages_flags(instance);
        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        OPENVINO_ASSERT(rt_params != nullptr);
        const bool probe_enabled = is_cm_pa_exec_probe_enabled();
        const bool kv_detail_probe_enabled = probe_enabled && is_cm_pa_kv_detail_probe_enabled();
        const auto execute_begin = probe_clock::now();
        double kv_cache_update_ms = 0.0;
        double kv_execute_stage_call_ms = 0.0;
        double kv_event_wait_ms = 0.0;
        double kv_event_submission_ms = 0.0;
        double kv_event_starting_ms = 0.0;
        double kv_event_executing_ms = 0.0;
        double kv_event_profile_total_ms = 0.0;
        double prepare_multi_mapping_ms = 0.0;
        double prepare_single_ids_ms = 0.0;

        GPU_DEBUG_TRACE_DETAIL << "ov::intel_gpu::cm::PagedAttentionCmImpl::execute():  stage = " << static_cast<int>(rt_params->stage) << std::endl;
        std::vector<event::ptr> res_event = events;
        const auto kv_begin = probe_clock::now();
        res_event = {execute_stage(res_event, instance, kv_cache_update)};
        if (probe_enabled) {
            kv_execute_stage_call_ms = std::chrono::duration<double, std::milli>(probe_clock::now() - kv_begin).count();
            kv_cache_update_ms = kv_execute_stage_call_ms;
        }

        if (kv_detail_probe_enabled && !res_event.empty() && res_event[0] != nullptr) {
            const auto kv_wait_begin = probe_clock::now();
            res_event[0]->wait();
            kv_event_wait_ms = std::chrono::duration<double, std::milli>(probe_clock::now() - kv_wait_begin).count();

            const auto profiling_info = res_event[0]->get_profiling_info();
            for (const auto& interval : profiling_info) {
                const double ms = std::chrono::duration<double, std::milli>(interval.value->value()).count();
                switch (interval.stage) {
                    case cldnn::instrumentation::profiling_stage::submission:
                        kv_event_submission_ms = ms;
                        break;
                    case cldnn::instrumentation::profiling_stage::starting:
                        kv_event_starting_ms = ms;
                        break;
                    case cldnn::instrumentation::profiling_stage::executing:
                        kv_event_executing_ms = ms;
                        break;
                    default:
                        break;
                }
            }
            kv_event_profile_total_ms = kv_event_submission_ms + kv_event_starting_ms + kv_event_executing_ms;
        }

        const auto execute_multi_token_path = [&]() {
            if (rt_params->multi_token_wg_count == 0) {
                return;
            }

            const auto prep_multi_begin = probe_clock::now();
            prepare_multi_token_mapping(instance);
            if (probe_enabled) {
                prepare_multi_mapping_ms += std::chrono::duration<double, std::milli>(probe_clock::now() - prep_multi_begin).count();
            }
            const bool xattn_enabled = desc->has_xattention && rt_params->enable_xattn_estimation;
            const bool is_xattn_bypassed = xattn_enabled ? bypass_xattn(params) : true;
            if (is_xattn_bypassed || !xattn_enabled) {
                GPU_DEBUG_TRACE_DETAIL << "Execute multi-token stage w/o XAttention estimation stages." << std::endl;

                res_event = {execute_stage(res_event, instance, *pa_multi_token_1)};
            } else {
                GPU_DEBUG_TRACE_DETAIL << "Execute multi-token stage w/ XAttention estimation stages." << std::endl;

                OPENVINO_ASSERT(rt_params->xattn_block_size == 128 || rt_params->xattn_block_size == 256,
                                "Unsupported xattention block size for multi token stage: ",
                                rt_params->xattn_block_size);

                const bool use_256 = rt_params->xattn_block_size == 256;
                Stage::Ptr& xattn_gemmqk = use_256 ? xattn_estimate_gemmqk_256 : xattn_estimate_gemmqk;
                Stage::Ptr& xattn_find_block = use_256 ? xattn_estimate_find_block_256 : xattn_estimate_find_block;
                Stage::Ptr& xattn_post_proc = use_256 ? xattn_estimate_post_proc_256 : xattn_estimate_post_proc;
                Stage::Ptr& pa_multi_token = use_256 ? pa_multi_token_256 : pa_multi_token_128;

                res_event = {execute_stage(res_event, instance, xattn_gemmqk)};
                XATTN_DUMP(instance, PagedAttentionInternBuffIdx::XATTN_GEMMQK_MAX);  // 2: kq_max_wg
                XATTN_DUMP(instance,
                           PagedAttentionInternBuffIdx::XATTN_GEMMQK_EXPSUMS);  // idx 3: kq_exp_partial_sum is subject to change in find_block kernel.
                res_event = {execute_stage(res_event, instance, xattn_find_block)};
                XATTN_DUMP(instance, PagedAttentionInternBuffIdx::XATTN_BLOCKMASK);  // 4: sparse_block_mask
#if FIND_DEBUG_ACC
                XATTN_DUMP(instance, PagedAttentionInternBuffIdx::XATTN_FIND_DEBUG_ACC);  // 6: kq_sum for debug purpose only
#endif
                res_event = {execute_stage(res_event, instance, xattn_post_proc)};
                res_event = {execute_stage(res_event, instance, pa_multi_token)};
            }
        };

        if (rt_params->stage == PagedAttentionStage::PREFILL) {
            execute_multi_token_path();
        } else if (rt_params->stage == PagedAttentionStage::MIXED) {
            if (rt_params->mixed_route_mode == MixedRouteMode::SPLIT) {
                const auto prep_single_begin = probe_clock::now();
                prepare_single_token_selected_ids(instance);
                if (probe_enabled) {
                    prepare_single_ids_ms += std::chrono::duration<double, std::milli>(probe_clock::now() - prep_single_begin).count();
                }
                if (rt_params->single_token_selected_count > 0) {
                    res_event = {execute_stage(res_event, instance, pa_single_token)};
                    res_event = {execute_stage(res_event, instance, pa_single_token_finalization)};
                }
                execute_multi_token_path();
            } else {
                execute_multi_token_path();
            }
        } else {
            const auto prep_single_begin = probe_clock::now();
            prepare_single_token_selected_ids(instance);
            if (probe_enabled) {
                prepare_single_ids_ms += std::chrono::duration<double, std::milli>(probe_clock::now() - prep_single_begin).count();
            }
            res_event = {execute_stage(res_event, instance, pa_single_token)};
            res_event = {execute_stage(res_event, instance, pa_single_token_finalization)};
        }

        if (probe_enabled) {
            const double execute_total_ms = std::chrono::duration<double, std::milli>(probe_clock::now() - execute_begin).count();
            std::fprintf(stderr,
                         "[CM_PA_EXEC_PROBE] stage=%d total_ms=%.6f kv_cache_update_ms=%.6f kv_execute_stage_call_ms=%.6f kv_event_wait_ms=%.6f kv_event_submission_ms=%.6f kv_event_starting_ms=%.6f kv_event_executing_ms=%.6f kv_event_profile_total_ms=%.6f prepare_multi_mapping_ms=%.6f prepare_single_ids_ms=%.6f multi_token_wg_count=%zu single_token_selected_count=%zu\n",
                         static_cast<int>(rt_params->stage),
                         execute_total_ms,
                         kv_cache_update_ms,
                         kv_execute_stage_call_ms,
                         kv_event_wait_ms,
                         kv_event_submission_ms,
                         kv_event_starting_ms,
                         kv_event_executing_ms,
                         kv_event_profile_total_ms,
                         prepare_multi_mapping_ms,
                         prepare_single_ids_ms,
                         rt_params->multi_token_wg_count,
                         rt_params->single_token_selected_count);
        }

        return res_event[0];
    }

    bool requires_update(primitive_inst& inst, const kernel_impl_params& impl_params) const override {
        const auto stage = get_paged_attention_stage(impl_params);

        // In case of MIXED mode execution Paged Attention may require dispatch data update and internal
        // buffers reallocation even if the input shapes haven't been changed. Therefore, check the current execution
        // mode and update parameters if needed
        return stage == PagedAttentionStage::MIXED ||
               ((stage == PagedAttentionStage::PREFILL || stage == PagedAttentionStage::UNKNOWN) && has_multiple_subsequences(impl_params));
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        std::vector<BufferDescriptor> internal_buffers;
        const bool lockable_mapping = is_cm_pa_force_lockable_mapping_enabled();

        const auto desc = params.typed_desc<paged_attention>();
        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        // Assume rt_params are updated, because get_internal_buffer_descs surely occurs after update_rt_params.
        OPENVINO_ASSERT(rt_params != nullptr);

        const auto stage = rt_params->stage;
        GPU_DEBUG_TRACE_DETAIL << " stage = " << static_cast<int>(stage) << std::endl;
        if (stage == PagedAttentionStage::GENERATE) {
            OPENVINO_ASSERT(rt_params->num_of_partitions != 0);
            size_t num_of_partitions = rt_params->num_of_partitions;

            const auto& input = params.input_layouts[0];
            const int64_t total_tokens = input.get_partial_shape()[0].get_length();
            auto buf_elements_count = static_cast<int64_t>(total_tokens * desc->heads_num * num_of_partitions);
            auto tmp_out_elements_count = static_cast<int64_t>(total_tokens * desc->heads_num * desc->v_head_size * num_of_partitions);

            internal_buffers.emplace_back(tmp_out_elements_count, ov::element::f32);  // 0: intermediate partition output
            internal_buffers.emplace_back(buf_elements_count, ov::element::f32);      // 1: softmax exp_sums
            internal_buffers.emplace_back(2, ov::element::i32, lockable_mapping);                 // 2: unused multi-token mapping placeholder
            internal_buffers.emplace_back(total_tokens, ov::element::i32, lockable_mapping);      // 3: selected sequence ids

            GPU_DEBUG_TRACE_DETAIL << "  internal buffer sizes: tmp_out=" << tmp_out_elements_count * 4 << "  exp_sums=" << buf_elements_count * 4 << std::endl;
        } else {
            int64_t decode_tmp_out_elements_count = 16;
            int64_t decode_buf_elements_count = 16;
            const bool needs_single_token_buffers = stage == PagedAttentionStage::MIXED &&
                                                    rt_params->mixed_route_mode == MixedRouteMode::SPLIT &&
                                                    rt_params->single_token_selected_count > 0;
            if (needs_single_token_buffers) {
                OPENVINO_ASSERT(rt_params->num_of_partitions != 0);
                decode_buf_elements_count = static_cast<int64_t>(rt_params->single_token_selected_count * desc->heads_num * rt_params->num_of_partitions);
                decode_tmp_out_elements_count = static_cast<int64_t>(rt_params->single_token_selected_count * desc->heads_num * desc->v_head_size * rt_params->num_of_partitions);
            }

            internal_buffers.emplace_back(decode_tmp_out_elements_count, ov::element::f32);  // 0: intermediate partition output
            internal_buffers.emplace_back(decode_buf_elements_count, ov::element::f32);       // 1: softmax exp_sums
            internal_buffers.emplace_back(std::max<int64_t>(2, static_cast<int64_t>(rt_params->multi_token_wg_count * 2)), ov::element::i32, lockable_mapping);  // 2: multi-token mapping
            internal_buffers.emplace_back(std::max<int64_t>(1, static_cast<int64_t>(rt_params->batch_size_in_sequences)), ov::element::i32, lockable_mapping);  // 3: selected ids / placeholder

            // internal buffer for XAttention
            if (rt_params->enable_xattn_estimation) {
                auto count_kq_max_wg = static_cast<int64_t>(desc->heads_num * rt_params->N_kq_groups * rt_params->q_stride_pad);
                internal_buffers.emplace_back(count_kq_max_wg, ov::element::f32);  // 4: kq_max_wg

                auto count_kq_exp_partial_sum = static_cast<int64_t>(desc->heads_num * rt_params->q_stride_pad * rt_params->k_block_pad);
                internal_buffers.emplace_back(count_kq_exp_partial_sum, ov::element::f32);  // 5: kq_exp_partial_sum

                auto count_elements_mask = static_cast<int64_t>(desc->heads_num * rt_params->q_block_pad * rt_params->k_block_pad);
                internal_buffers.emplace_back(count_elements_mask, ov::element::boolean);  // 6: sparse_block_mask

                auto count_elements_mask_merged = static_cast<int64_t>(desc->heads_num * rt_params->q_block_pad_merged * rt_params->k_block_pad);
                internal_buffers.emplace_back(count_elements_mask_merged, ov::element::boolean);  // 7: sparse_block_mask_wg

#if FIND_DEBUG_ACC
                const size_t sum_per_n_token_in_block = static_cast<size_t>(rt_params->xattn_block_size / STRIDE);
                size_t q_block_input = rt_params->q_stride_pad / sum_per_n_token_in_block;
                auto count_elements_kq_sum = static_cast<int64_t>(desc->heads_num * q_block_input * rt_params->k_block_pad);
                internal_buffers.emplace_back(count_elements_kq_sum, ov::element::f16);  // 8: kq_sum
#endif

                GPU_DEBUG_TRACE_DETAIL << "  internal buffer sizes: count_kq_max_wg=" << count_kq_max_wg * 4
                                       << "  count_kq_exp_partial_sum=" << count_kq_exp_partial_sum * 4 << "  count_elements_mask=" << count_elements_mask * 1
                                       << "  count_elements_mask_merged=" << count_elements_mask_merged * 1 << std::endl;
            }
        }

        return internal_buffers;
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<PagedAttentionCmImpl>(this);
    }

private:
    void validate_xattn_inputs(const kernel_impl_params& params, size_t batch_size) {
        const auto& input_mem = params.memory_deps;

        auto validate_input = [&](size_t idx, const char* name) {
            const auto it = input_mem.find(idx);
            if (it == input_mem.end() || it->second == nullptr)
                OPENVINO_THROW("XAttention ", name, " input is required at index ", idx);

            const auto input_size = it->second->count();
            if (input_size != 1 && input_size != batch_size)
                OPENVINO_THROW("XAttention ", name, " input size (", input_size,
                               ") must be 1 or equal to batch size (", batch_size, ")");
        };

        validate_input(PagedAttentionInputIdx::XATTENTION_BLOCK_SIZE, "block size");
        validate_input(PagedAttentionInputIdx::XATTENTION_THRESHOLD, "threshold");
    }

    size_t get_xattn_block_size(const kernel_impl_params& params, const size_t seq_idx = 0) {
        constexpr int32_t block_size_128 = 128;
        constexpr int32_t block_size_256 = 256;

        const auto rt_params = static_cast<const PagedAttentionRuntimeParams*>(m_rt_params.get());
        OPENVINO_ASSERT(rt_params != nullptr, "PagedAttention runtime params are not initialized");
        OPENVINO_ASSERT(rt_params->enable_xattn_estimation,
                        "XAttention block size must be accessed only when enable_xattn_estimation is true");

        const auto desc = params.typed_desc<paged_attention>();
        OPENVINO_ASSERT(desc->has_xattention,
                        "XAttention block size must be accessed only when has_xattention is true");

        const auto& input_mem = params.memory_deps;
        const auto it = input_mem.find(PagedAttentionInputIdx::XATTENTION_BLOCK_SIZE);
        if (it == input_mem.end() || it->second == nullptr) {
            OPENVINO_THROW("XAttention block size input is required at index ", static_cast<size_t>(PagedAttentionInputIdx::XATTENTION_BLOCK_SIZE));
        }

        mem_lock<int32_t, mem_lock_type::read> lock(it->second, *params.strm);
        if (seq_idx >= lock.size()) {
            OPENVINO_THROW("XAttention block size input index out of range: seq_idx=", seq_idx, ", input_size=", lock.size());
        }

        int32_t xattn_block_size = static_cast<int32_t>(lock[seq_idx]);
        GPU_DEBUG_TRACE_DETAIL << "XAttention block size from input: " << xattn_block_size << std::endl;

        if (params.get_device_info().arch < gpu_arch::xe2) {
            return block_size_128;
        }

        xattn_block_size = (xattn_block_size == block_size_128 || xattn_block_size == block_size_256) ? xattn_block_size : block_size_256;
        return xattn_block_size;
    }

#if DUMP_XATTN_INTERNALS
    void dump_xattn_internals(primitive_inst& instance, PagedAttentionInternBuffIdx idx) {
        const char* dump_root_env = std::getenv("DUMP_XATTN_INTERNALS");
        if (dump_root_env == nullptr || dump_root_env[0] == '\0') {
            return;  // skip dumping
        }

        cldnn::stream& stream = instance.get_network().get_stream();
        stream.finish();

        const auto node_name = instance.get_node().id();
        auto output_mem = instance.get_intermediates_memories()[idx];
        mem_lock<char, mem_lock_type::read> lock(output_mem, stream);
        auto& layout = output_mem->get_layout();
        std::string data_type = ov::element::Type(layout.data_type).get_type_name();
        std::string format = layout.format.to_string();
        std::string tensor;
        auto dims = layout.get_dims();
        for (size_t r = 0; r < layout.get_rank(); r++) {
            tensor += ("_" + to_string(dims[r]));
        }

        std::string out_path =
            std::string(dump_root_env) + "xattn_internals_" + std::to_string(idx) + "__" + node_name + "__" + data_type + "_" + tensor + "__" + format + ".bin";
        try {
            ov::util::save_binary(out_path, lock.data(), output_mem->size());
            std::cout << "[dump_xattn_internals] dump to " << out_path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[dump_xattn_internals] Failed to save dump to '" << out_path << "': " << e.what() << "\n";
        }
    }
#endif
};

std::unique_ptr<primitive_impl> PagedAttentionImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<paged_attention>());
    try {
        return std::make_unique<PagedAttentionCmImpl>(params);
    } catch (const std::exception& e) {
        OPENVINO_THROW("Failed to create PagedAttentionCmImpl: ", e.what());
    }
}

}  // namespace ov::intel_gpu::cm
// BIND_BINARY_BUFFER_WITH_TYPE(cldnn::paged_attention)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::PagedAttentionCmImpl)
