// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

#include "common_utils/jitter.hpp"
#include "common_utils/kernel_generator_base.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "kv_cache_inst.h"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/float16.hpp"
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
constexpr size_t kTurboQuantBits = 4;
constexpr size_t kTurboQuantCentroidsCount = static_cast<size_t>(1u << kTurboQuantBits);
constexpr size_t kTurboQuantBoundariesCount = kTurboQuantCentroidsCount - 1;

bool is_turboquant_requested(const kernel_impl_params& params) {
    const auto mode = params.get_program().get_config().get_key_cache_quant_mode();
    return mode == ov::internal::CacheQuantMode::TURBOQUANT;
}

bool has_turboquant_kcache_layout(const kernel_impl_params& params) {
    const auto desc = params.typed_desc<paged_attention>();
    const auto& key_cache_layout = params.input_layouts[PagedAttentionInputIdx::KEY_CACHE];

    if (!data_type_traits::is_i8_u8(key_cache_layout.data_type)) {
        return false;
    }

    const auto shape = key_cache_layout.get_shape();
    if (shape.size() < 4) {
        return false;
    }

    const size_t packed_bytes = (desc->k_head_size * 4 + 7) / 8;
    const size_t expected_last_dim = packed_bytes + 2;
    return shape[3] == expected_last_dim;
}

bool use_tq_prefill_debug_fallback_path() {
    const char* env = std::getenv("OV_GPU_TQ_PREFILL_EXTRA_PATH");
    if (env == nullptr || env[0] == '\0')
        return false;

    std::string v(env);
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    return v == "1" || v == "on" || v == "true" || v == "yes";
}

void reserve_turboquant_index_gap(std::vector<BufferDescriptor>& internal_buffers) {
    // Reserve 2..5 indices used by XAttention buffers so TurboQuant buffers keep fixed indices.
    internal_buffers.emplace_back(16, ov::element::f32);      // 2: placeholder
    internal_buffers.emplace_back(16, ov::element::f32);      // 3: placeholder
    internal_buffers.emplace_back(16, ov::element::boolean);  // 4: placeholder
    internal_buffers.emplace_back(16, ov::element::boolean);  // 5: placeholder
#if FIND_DEBUG_ACC
    internal_buffers.emplace_back(16, ov::element::f16);  // 6: placeholder for XATTN_FIND_DEBUG_ACC
#endif
}

void append_turboquant_internal_buffers(std::vector<BufferDescriptor>& internal_buffers, const kernel_impl_params& params) {
    const auto desc = params.typed_desc<paged_attention>();
    const auto q_t_elements = desc->k_head_size * desc->k_head_size;
    const bool lockable = true;
    internal_buffers.emplace_back(q_t_elements, ov::element::f16, lockable);  // TQ_Q_TRANSFORM
    internal_buffers.emplace_back(kTurboQuantCentroidsCount, ov::element::f16, lockable);   // TQ_CENTROIDS
    internal_buffers.emplace_back(kTurboQuantBoundariesCount, ov::element::f16, lockable);  // TQ_BOUNDARIES
}
}  // namespace

class PagedAttentionCmImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::PagedAttentionCmImpl)

    Stage::Ptr kv_cache_update;
    Stage::Ptr kv_cache_update_no_tq_tmp;
    Stage::Ptr pa_single_token;
    Stage::Ptr pa_single_token_finalization = make_stage<PagedAttentionGeneratorSingleTokenFinalization>();
    Stage::Ptr pa_multi_token_1;
    Stage::Ptr pa_multi_token_1_no_tq_tmp;
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
        m_use_turboquant = is_turboquant_requested(params) || has_turboquant_kcache_layout(params);
        GPU_DEBUG_TRACE_DETAIL << "[CM PA] ctor turboquant detection: use_turboquant=" << static_cast<int>(m_use_turboquant)
                               << ", by_config=" << static_cast<int>(is_turboquant_requested(params))
                               << ", by_layout=" << static_cast<int>(has_turboquant_kcache_layout(params)) << std::endl;

        if (m_use_turboquant) {
            if (params.get_device_info().arch < gpu_arch::xe2) {
                OPENVINO_THROW("TurboQuant is not supported on pre-Xe2 GPU architecture");
            }
            if (desc->has_xattention) {
                OPENVINO_THROW("TurboQuant is not supported with XAttention");
            }
        }

        kv_cache_update = make_stage<PagedAttentionGeneratorKVCacheUpdate>(m_use_turboquant);
        kv_cache_update_no_tq_tmp = make_stage<PagedAttentionGeneratorKVCacheUpdate>(false, true);
        pa_single_token = make_stage<PagedAttentionGeneratorSingleToken>(m_use_turboquant);
        pa_multi_token_1 = make_stage<PagedAttentionGeneratorMultiToken>(1, m_use_turboquant);
        pa_multi_token_1_no_tq_tmp = make_stage<PagedAttentionGeneratorMultiToken>(1, false, true);

        GPU_DEBUG_TRACE_DETAIL << "ov::intel_gpu::cm::PagedAttentionCmImpl::PagedAttentionCmImpl()" << std::endl;
        add_stage(kv_cache_update, params);
        add_stage(pa_single_token, params);
        add_stage(pa_single_token_finalization, params);
        add_stage(pa_multi_token_1, params);
        if (m_use_turboquant && use_tq_prefill_debug_fallback_path()) {
            add_stage(kv_cache_update_no_tq_tmp, params);
            add_stage(pa_multi_token_1_no_tq_tmp, params);
            GPU_DEBUG_TRACE_DETAIL << "[CM PA][TQ] OV_GPU_TQ_PREFILL_EXTRA_PATH enabled: fallback non-TQ prefill stages are added." << std::endl;
        }
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

        const bool turboquant_by_config = is_turboquant_requested(params);
        const bool turboquant_by_layout = has_turboquant_kcache_layout(params);
        const bool runtime_use_turboquant = turboquant_by_config || turboquant_by_layout;
        if (m_use_turboquant != runtime_use_turboquant) {
            GPU_DEBUG_TRACE_DETAIL << "[CM PA] update_rt_params turboquant transition: " << static_cast<int>(m_use_turboquant)
                                   << " -> " << static_cast<int>(runtime_use_turboquant)
                                   << " (by_config=" << static_cast<int>(turboquant_by_config)
                                   << ", by_layout=" << static_cast<int>(turboquant_by_layout) << ")" << std::endl;
            if (!runtime_use_turboquant) {
                m_tq_tables_initialized = false;
                m_tq_tables_head_size = 0;
            }
        }
        m_use_turboquant = runtime_use_turboquant;

        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        const auto& desc = params.typed_desc<paged_attention>();

        rt_params->stage = get_paged_attention_stage(params);
        const auto max_context_len = get_max_context_len(params);
        rt_params->max_context_len = max_context_len;
        GPU_DEBUG_TRACE_DETAIL << "update_rt_params for stage: " << static_cast<size_t>(rt_params->stage) << "  max_context_len: " << rt_params->max_context_len
                               << std::endl;

        if (rt_params->stage == PagedAttentionStage::GENERATE) {
            auto partition_size = PagedAttentionGeneratorSingleToken::get_partition_size(desc->has_xattention);
            rt_params->num_of_partitions = ceil_div(max_context_len, partition_size);
            rt_params->q_chunking = get_single_token_q_chunking(params, *desc, partition_size);
            GPU_DEBUG_TRACE_DETAIL << "  partition_size: " << partition_size << "  num_of_partitions: " << rt_params->num_of_partitions << std::endl;
        } else {
            if (desc->has_xattention) {
                update_xattn_rt_params(params);
            } else {
                rt_params->xattn_block_size = 1;  // disable xattn for pa
            }
        }
    }

    // update impl_parameter and rt_parameter
    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        PrimitiveImplCM::update(inst, impl_params);
        update_rt_params(inst);
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        const auto& params = *instance.get_impl_params();
        const auto desc = params.typed_desc<paged_attention>();
        auto& stream = instance.get_network().get_stream();

        stream.finish();

        if (m_use_turboquant) {
            if (params.get_device_info().arch < gpu_arch::xe2) {
                OPENVINO_THROW("TurboQuant is not supported on pre-Xe2 GPU architecture");
            }
            if (desc->has_xattention) {
                OPENVINO_THROW("TurboQuant is not supported with XAttention");
            }
            const auto key_cache_layout = params.input_layouts[PagedAttentionInputIdx::KEY_CACHE];
            OPENVINO_ASSERT(data_type_traits::is_i8_u8(key_cache_layout.data_type),
                            "TurboQuant requires i8/u8 key cache precision");
            OPENVINO_ASSERT(!desc->is_key_by_channel,
                            "TurboQuant requires key cache quantization mode BY_TOKEN/TURBOQUANT");
        }

        update_stages_flags(instance);
        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        OPENVINO_ASSERT(rt_params != nullptr);

        const bool turboquant_by_config = is_turboquant_requested(params);
        const bool turboquant_by_layout = has_turboquant_kcache_layout(params);
        GPU_DEBUG_TRACE_DETAIL << "ov::intel_gpu::cm::PagedAttentionCmImpl::execute():  stage = " << static_cast<int>(rt_params->stage)
                       << ", use_turboquant = " << static_cast<int>(m_use_turboquant)
                       << ", by_config = " << static_cast<int>(turboquant_by_config)
                       << ", by_layout = " << static_cast<int>(turboquant_by_layout)
                       << std::endl;
        std::vector<event::ptr> res_event = events;
        if (m_use_turboquant) {
            auto& intermediates = instance.get_intermediates_memories();
            const size_t head_size = desc->k_head_size;
            if (!m_tq_tables_initialized || m_tq_tables_head_size != head_size) {
                GPU_DEBUG_TRACE_DETAIL << "[CM PA][TQ] init tables: begin (head_size=" << head_size
                                       << ", prev_initialized=" << static_cast<int>(m_tq_tables_initialized)
                                       << ", prev_head_size=" << m_tq_tables_head_size << ")" << std::endl;
                OPENVINO_ASSERT(static_cast<size_t>(PagedAttentionInternBuffIdx::TQ_Q_TRANSFORM) < intermediates.size(),
                                "TurboQuant internal buffer index is out of range for TQ_Q_TRANSFORM");
                OPENVINO_ASSERT(static_cast<size_t>(PagedAttentionInternBuffIdx::TQ_CENTROIDS) < intermediates.size(),
                                "TurboQuant internal buffer index is out of range for TQ_CENTROIDS");
                OPENVINO_ASSERT(static_cast<size_t>(PagedAttentionInternBuffIdx::TQ_BOUNDARIES) < intermediates.size(),
                                "TurboQuant internal buffer index is out of range for TQ_BOUNDARIES");

                const auto& q_t_mem = intermediates[PagedAttentionInternBuffIdx::TQ_Q_TRANSFORM];
                const auto& centroids_mem = intermediates[PagedAttentionInternBuffIdx::TQ_CENTROIDS];
                const auto& boundaries_mem = intermediates[PagedAttentionInternBuffIdx::TQ_BOUNDARIES];

                OPENVINO_ASSERT(q_t_mem != nullptr && centroids_mem != nullptr && boundaries_mem != nullptr,
                                "TurboQuant internal buffers are not allocated");

                const size_t q_t_count = q_t_mem->size() / sizeof(ov::float16);
                const size_t centroids_count = centroids_mem->size() / sizeof(ov::float16);
                const size_t boundaries_count = boundaries_mem->size() / sizeof(ov::float16);

                OPENVINO_ASSERT(q_t_count == head_size * head_size,
                                "Unexpected TurboQuant Q_T size: expected ",
                                head_size * head_size,
                                ", got ",
                                q_t_count);
                OPENVINO_ASSERT(centroids_count == kTurboQuantCentroidsCount,
                                "Unexpected TurboQuant centroids size: expected ",
                                kTurboQuantCentroidsCount,
                                ", got ",
                                centroids_count);
                OPENVINO_ASSERT(boundaries_count == kTurboQuantBoundariesCount,
                                "Unexpected TurboQuant boundaries size: expected ",
                                kTurboQuantBoundariesCount,
                                ", got ",
                                boundaries_count);

                GPU_DEBUG_TRACE_DETAIL << "[CM PA][TQ] table buffer sizes (elements): q_t=" << q_t_count
                                       << ", centroids=" << centroids_count
                                       << ", boundaries=" << boundaries_count << std::endl;

                std::vector<ov::float16> q_t_host(q_t_count);
                std::vector<ov::float16> centroids_host(centroids_count);
                std::vector<ov::float16> boundaries_host(boundaries_count);

                // Internal TurboQuant tables. Keep table ownership inside primitive internals.
                for (size_t i = 0; i < head_size; ++i) {
                    for (size_t j = 0; j < head_size; ++j) {
                        q_t_host[i * head_size + j] = ov::float16(i == j ? 1.0f : 0.0f);
                    }
                }

                for (size_t i = 0; i < centroids_count; ++i) {
                    const float c = -1.0f + (2.0f * static_cast<float>(i)) / static_cast<float>(centroids_count - 1);
                    centroids_host[i] = ov::float16(c);
                }

                for (size_t i = 0; i < boundaries_count; ++i) {
                    const float c0 = -1.0f + (2.0f * static_cast<float>(i)) / static_cast<float>(centroids_count - 1);
                    const float c1 = -1.0f + (2.0f * static_cast<float>(i + 1)) / static_cast<float>(centroids_count - 1);
                    boundaries_host[i] = ov::float16(0.5f * (c0 + c1));
                }

                q_t_mem->copy_from(stream, q_t_host.data(), true);
                centroids_mem->copy_from(stream, centroids_host.data(), true);
                boundaries_mem->copy_from(stream, boundaries_host.data(), true);

                m_tq_tables_initialized = true;
                m_tq_tables_head_size = head_size;
                GPU_DEBUG_TRACE_DETAIL << "[CM PA][TQ] init tables: done" << std::endl;
            }
        }

        OPENVINO_ASSERT(kv_cache_update != nullptr, "kv_cache_update stage is null before execute_stage");
        GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage begin: kv_cache_update, stage_ptr=" << kv_cache_update.get() << std::endl;
        res_event = {execute_stage(res_event, instance, kv_cache_update)};
        GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage end: kv_cache_update" << std::endl;
        stream.finish();

        if (rt_params->stage == PagedAttentionStage::PREFILL || rt_params->stage == PagedAttentionStage::MIXED) {
            if (m_use_turboquant || desc->has_xattention == false || bypass_xattn(params)) {
                GPU_DEBUG_TRACE_DETAIL << "Execute multi-token stage w/o XAttention estimation stages." << std::endl;
                const bool use_extra_path = m_use_turboquant && use_tq_prefill_debug_fallback_path();
                if (use_extra_path) {
                    GPU_DEBUG_TRACE_DETAIL << "[CM PA][TQ] OV_GPU_TQ_PREFILL_EXTRA_PATH enabled: bypass pa_multi_token_turboquant, use temporary KV fallback path." << std::endl;
                    GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage begin: kv_cache_update_no_tq_tmp" << std::endl;
                    res_event = {execute_stage(res_event, instance, *kv_cache_update_no_tq_tmp)};
                    GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage end: kv_cache_update_no_tq_tmp" << std::endl;
                    stream.finish();

                    GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage begin: pa_multi_token_1_no_tq_tmp" << std::endl;
                    res_event = {execute_stage(res_event, instance, *pa_multi_token_1_no_tq_tmp)};
                    GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage end: pa_multi_token_1_no_tq_tmp" << std::endl;
                } else {
                    GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage begin: pa_multi_token_1" << std::endl;
                    res_event = {execute_stage(res_event, instance, *pa_multi_token_1)};
                    GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage end: pa_multi_token_1" << std::endl;
                }
                stream.finish();
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

                GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage begin: xattn_gemmqk" << std::endl;
                res_event = {execute_stage(res_event, instance, xattn_gemmqk)};
                GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage end: xattn_gemmqk" << std::endl;
                stream.finish();
                XATTN_DUMP(instance, PagedAttentionInternBuffIdx::XATTN_GEMMQK_MAX);  // 2: kq_max_wg
                XATTN_DUMP(instance,
                           PagedAttentionInternBuffIdx::XATTN_GEMMQK_EXPSUMS);  // idx 3: kq_exp_partial_sum is subject to change in find_block kernel.
                GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage begin: xattn_find_block" << std::endl;
                res_event = {execute_stage(res_event, instance, xattn_find_block)};
                GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage end: xattn_find_block" << std::endl;
                stream.finish();
                XATTN_DUMP(instance, PagedAttentionInternBuffIdx::XATTN_BLOCKMASK);  // 4: sparse_block_mask
#if FIND_DEBUG_ACC
                XATTN_DUMP(instance, PagedAttentionInternBuffIdx::XATTN_FIND_DEBUG_ACC);  // 6: kq_sum for debug purpose only
#endif
                GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage begin: xattn_post_proc" << std::endl;
                res_event = {execute_stage(res_event, instance, xattn_post_proc)};
                GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage end: xattn_post_proc" << std::endl;
                stream.finish();
                GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage begin: pa_multi_token" << std::endl;
                res_event = {execute_stage(res_event, instance, pa_multi_token)};
                GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage end: pa_multi_token" << std::endl;
                stream.finish();
            }
        } else {
            GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage begin: pa_single_token" << std::endl;
            res_event = {execute_stage(res_event, instance, pa_single_token)};
            GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage end: pa_single_token" << std::endl;
            stream.finish();
            GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage begin: pa_single_token_finalization" << std::endl;
            res_event = {execute_stage(res_event, instance, pa_single_token_finalization)};
            GPU_DEBUG_TRACE_DETAIL << "[CM PA] execute_stage end: pa_single_token_finalization" << std::endl;
            stream.finish();
        }
        return res_event[0];
    }

    bool requires_update(primitive_inst& inst, const kernel_impl_params& impl_params) const override {
        const auto stage = get_paged_attention_stage(impl_params);

        // In case of MIXED mode execution Paged Attention may require dispatch data update and internal
        // buffers reallocation even if the input shapes haven't been changed. Therefore, check the current execution
        // mode and update parameters if needed
        return stage == PagedAttentionStage::MIXED;
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        std::vector<BufferDescriptor> internal_buffers;

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

            if (m_use_turboquant) {
                reserve_turboquant_index_gap(internal_buffers);
                append_turboquant_internal_buffers(internal_buffers, params);
            }

            GPU_DEBUG_TRACE_DETAIL << "  internal buffer sizes: tmp_out=" << tmp_out_elements_count * 4 << "  exp_sums=" << buf_elements_count * 4 << std::endl;
        } else {
            internal_buffers.emplace_back(16, ov::element::f32);  // 0: intermediate partition output
            internal_buffers.emplace_back(16, ov::element::f32);  // 1: softmax exp_sums

            // internal buffer for XAttention
            if (desc->has_xattention) {
                auto count_kq_max_wg = static_cast<int64_t>(desc->heads_num * rt_params->N_kq_groups * rt_params->q_stride_pad);
                internal_buffers.emplace_back(count_kq_max_wg, ov::element::f32);  // 2: kq_max_wg

                auto count_kq_exp_partial_sum = static_cast<int64_t>(desc->heads_num * rt_params->q_stride_pad * rt_params->k_block_pad);
                internal_buffers.emplace_back(count_kq_exp_partial_sum, ov::element::f32);  // 3: kq_exp_partial_sum

                auto count_elements_mask = static_cast<int64_t>(desc->heads_num * rt_params->q_block_pad * rt_params->k_block_pad);
                internal_buffers.emplace_back(count_elements_mask, ov::element::boolean);  // 4: sparse_block_mask

                auto count_elements_mask_merged = static_cast<int64_t>(desc->heads_num * rt_params->q_block_pad_merged * rt_params->k_block_pad);
                internal_buffers.emplace_back(count_elements_mask_merged, ov::element::boolean);  // 5: sparse_block_mask_wg

#if FIND_DEBUG_ACC
                const size_t sum_per_n_token_in_block = static_cast<size_t>(rt_params->xattn_block_size / STRIDE);
                size_t q_block_input = rt_params->q_stride_pad / sum_per_n_token_in_block;
                auto count_elements_kq_sum = static_cast<int64_t>(desc->heads_num * q_block_input * rt_params->k_block_pad);
                internal_buffers.emplace_back(count_elements_kq_sum, ov::element::f16);  // 6: kq_sum
#endif

                GPU_DEBUG_TRACE_DETAIL << "  internal buffer sizes: count_kq_max_wg=" << count_kq_max_wg * 4
                                       << "  count_kq_exp_partial_sum=" << count_kq_exp_partial_sum * 4 << "  count_elements_mask=" << count_elements_mask * 1
                                       << "  count_elements_mask_merged=" << count_elements_mask_merged * 1 << std::endl;
            } else if (m_use_turboquant) {
                reserve_turboquant_index_gap(internal_buffers);
            }

            if (m_use_turboquant) {
                append_turboquant_internal_buffers(internal_buffers, params);
            }

            if (m_use_turboquant && use_tq_prefill_debug_fallback_path()) {
                OPENVINO_ASSERT(rt_params->max_context_len > 0, "Unexpected max_context_len for temporary KV buffers");
                const size_t block_size = PA_KV_CACHE_BLOCK_SIZE_LEGACY;
                const size_t num_blocks = ceil_div(rt_params->max_context_len, block_size);

                const size_t tmp_key_head_size = desc->k_head_size + 4;
                const size_t tmp_value_head_size = desc->v_head_size + 4;
                const size_t tmp_key_count = num_blocks * desc->kv_heads_num * block_size * tmp_key_head_size;
                const size_t tmp_value_count = num_blocks * desc->kv_heads_num * block_size * tmp_value_head_size;

                internal_buffers.emplace_back(tmp_key_count, ov::element::u8);    // TEMP_KEY_CACHE
                internal_buffers.emplace_back(tmp_value_count, ov::element::u8);  // TEMP_VALUE_CACHE

                GPU_DEBUG_TRACE_DETAIL << "[CM PA][TQ] temporary KV buffers: key_count=" << tmp_key_count
                                       << " value_count=" << tmp_value_count
                                       << " num_blocks=" << num_blocks << std::endl;
            }
        }

        return internal_buffers;
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        auto copy = std::make_unique<PagedAttentionCmImpl>();
        copy->m_use_turboquant = m_use_turboquant;
        copy->_order = _order;
        copy->m_rt_params = nullptr;
        copy->m_manager = m_manager;
        copy->can_reuse_memory = can_reuse_memory;
        copy->can_share_kernels = can_share_kernels;
        copy->_weights_reorder_params = _weights_reorder_params;
        copy->_kernel_name = _kernel_name;
        copy->_is_dynamic = _is_dynamic;

        copy->kv_cache_update = copy->make_stage<PagedAttentionGeneratorKVCacheUpdate>(copy->m_use_turboquant);
        copy->kv_cache_update_no_tq_tmp = copy->make_stage<PagedAttentionGeneratorKVCacheUpdate>(false, true);
        copy->pa_single_token = copy->make_stage<PagedAttentionGeneratorSingleToken>(copy->m_use_turboquant);
        copy->pa_multi_token_1 = copy->make_stage<PagedAttentionGeneratorMultiToken>(1, copy->m_use_turboquant);
        copy->pa_multi_token_1_no_tq_tmp = copy->make_stage<PagedAttentionGeneratorMultiToken>(1, false, true);

        OPENVINO_ASSERT(copy->_stages.size() == _stages.size(),
                        "PagedAttentionCmImpl clone stage count mismatch: expected ",
                        _stages.size(),
                        ", got ",
                        copy->_stages.size());

        for (size_t i = 0; i < copy->_stages.size(); i++) {
            copy->_stages[i]->kd = _stages[i]->kd;
            if (_stages[i]->kernel) {
                copy->_stages[i]->kernel = _stages[i]->kernel->clone();
            }
        }

        return copy;
    }

private:
    bool m_use_turboquant = false;
    bool m_tq_tables_initialized = false;
    size_t m_tq_tables_head_size = 0;

    // Get XAttention block size from input tensor.
    // If the input is not provided, throw exception.
    // If the input is not valid, return default block size based on GPU architecture.
    size_t get_xattn_block_size(const kernel_impl_params& params, const size_t seq_idx = 0) {
        constexpr int32_t block_size_128 = 128;
        constexpr int32_t block_size_256 = 256;

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
