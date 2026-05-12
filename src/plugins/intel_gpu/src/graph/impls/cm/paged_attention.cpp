// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "common_utils/jitter.hpp"
#include "common_utils/kernel_generator_base.hpp"
#include "include/xattn_subseq_meta.hpp"
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

// -----------------------------
// Input readers
// -----------------------------

std::vector<int32_t> read_i32_input(const kernel_impl_params& params, size_t input_idx) {
    const auto& memory_deps = params.memory_deps;
    const auto mem = memory_deps.at(input_idx);
    mem_lock<int32_t, mem_lock_type::read> lock(mem, *params.strm);
    return std::vector<int32_t>(lock.begin(), lock.end());
}

std::vector<int32_t> read_i32_input(const primitive_inst& instance, size_t input_idx) {
    auto mem = instance.input_memory_ptr(input_idx);
    mem_lock<int32_t, mem_lock_type::read> lock(mem, instance.get_network().get_stream());
    return std::vector<int32_t>(lock.begin(), lock.end());
}

// -----------------------------
// Routing/config helpers
// -----------------------------

MixedRouteMode get_mixed_route_mode_from_config(const kernel_impl_params& params) {
    std::string mode = params.get_program().get_config().get_pa_mixed_route_mode();
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (mode == "split") {
        return MixedRouteMode::SPLIT;
    }

    return MixedRouteMode::MULTI;
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
        m_mixed_route_mode = get_mixed_route_mode_from_config(params);

        GPU_DEBUG_TRACE_DETAIL << "ov::intel_gpu::cm::PagedAttentionCmImpl::PagedAttentionCmImpl()"
                               << " with mode: " << (m_mixed_route_mode == MixedRouteMode::SPLIT ? "split" : "multi") << std::endl;
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

    void update_xattn_rt_params(const primitive_inst& instance) {
        const auto& params = *instance.get_impl_params();
        const auto desc = params.typed_desc<paged_attention>();
        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());

        const size_t block_size = get_xattn_block_size(params);
        const uint32_t block_wg_n = XAttentionEstimateGeneratorBase::get_block_wg_n(params);
        const uint32_t block_wg_m = XAttentionEstimateGeneratorBase::get_block_wg_m(params);
        const size_t heads_num = desc->heads_num;
        const size_t merged_q_num = PagedAttentionGeneratorMultiToken::get_wg_seq_len(params) / block_size;
        const size_t sum_per_token_in_block = block_size / STRIDE;
        const size_t k_block_in_group = block_wg_n / sum_per_token_in_block;
        const size_t sizeof_softmax = sizeof(float);

        const auto subsequence_begins = read_i32_input(params, PagedAttentionInputIdx::SUBSEQUENCE_BEGINS);
        const auto past_lens = read_i32_input(params, PagedAttentionInputIdx::PAST_LENS);
        const auto block_indices_begins = read_i32_input(instance, PagedAttentionInputIdx::BLOCK_INDICES_BEGINS);

        const bool use_split_mixed = rt_params->stage == PagedAttentionStage::MIXED && m_mixed_route_mode == MixedRouteMode::SPLIT;

        size_t total_wg_count = 0;
        size_t max_q_block_pad = 0;
        size_t max_merged_q_blocks = 0;
        size_t cumul_kq_max_bytes = 0;
        size_t cumul_exp_sum_bytes = 0;
        size_t cumul_mask_elems = 0;
        size_t cumul_mask_wg_elems = 0;
        size_t num_xattn_subseqs = 0;

        m_xattn_meta.clear();
        m_xattn_find_wg_map.clear();
        m_xattn_post_wg_map.clear();
        m_xattn_meta.reserve((subsequence_begins.size() > 0 ? (subsequence_begins.size() - 1) : 0) * XATTN_META_STRIDE);
        m_xattn_find_wg_map.reserve((subsequence_begins.size() > 0 ? (subsequence_begins.size() - 1) : 0) * 2);
        m_xattn_post_wg_map.reserve((subsequence_begins.size() > 0 ? (subsequence_begins.size() - 1) : 0) * 2);

        auto to_i32_checked = [](size_t value, const char* field_name) {
            OPENVINO_ASSERT(value <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                            "XAttention metadata value is too large for int32 field ",
                            field_name,
                            ": ",
                            value);
            return static_cast<int32_t>(value);
        };

        for (size_t i = 0; i + 1 < subsequence_begins.size(); ++i) {
            const auto q_len = static_cast<size_t>(std::max<int32_t>(subsequence_begins[i + 1] - subsequence_begins[i], 0));
            if (q_len == 0)
                continue;

            if (use_split_mixed) {
                const auto past_len = static_cast<size_t>(std::max<int32_t>(past_lens[i], 0));
                const bool decode_subseq = (q_len == 1) && (past_len > 0);
                if (decode_subseq)
                    continue;
            }

            const auto past_len_s = static_cast<size_t>(std::max<int32_t>(past_lens[i], 0));
            const size_t kv_len = past_len_s + q_len;
            const size_t subseq_q_begin = static_cast<size_t>(subsequence_begins[i]);
            const int32_t bi_begin = (i < block_indices_begins.size()) ? block_indices_begins[i] : 0;

            const size_t M_s = q_len / STRIDE;
            const size_t N_s = kv_len / STRIDE;
            const size_t q_stride_pad_s = round_up_to(M_s, block_wg_m);
            const size_t N_kq_groups_s = ceil_div(N_s, block_wg_n);
            const size_t q_block_pad_s = ceil_div(q_len, block_size);
            const size_t k_block_pad_s = k_block_in_group * N_kq_groups_s;
            const size_t merged_q_blocks_s = ceil_div(q_block_pad_s, merged_q_num);
            const size_t k_block_s = ceil_div(kv_len, block_size);
            const size_t q_block_s = ceil_div(q_len, block_size);
            const size_t causal_start_s = k_block_s - q_block_s;
            const size_t q_start_strided_s = (kv_len - q_len) / STRIDE;
            const size_t wg_count_s = N_kq_groups_s * (q_stride_pad_s / block_wg_m);
            const int32_t subseq_id = to_i32_checked(num_xattn_subseqs, "subseq_id");

            m_xattn_meta.push_back(to_i32_checked(subseq_q_begin, "subseq_q_begin"));
            m_xattn_meta.push_back(to_i32_checked(q_len, "q_len"));
            m_xattn_meta.push_back(to_i32_checked(M_s, "M"));
            m_xattn_meta.push_back(to_i32_checked(N_s, "N"));
            m_xattn_meta.push_back(to_i32_checked(q_stride_pad_s, "q_stride_pad"));
            m_xattn_meta.push_back(to_i32_checked(N_kq_groups_s, "N_kq_groups"));
            m_xattn_meta.push_back(to_i32_checked(q_block_pad_s, "q_block_pad"));
            m_xattn_meta.push_back(to_i32_checked(k_block_pad_s, "k_block_pad"));
            m_xattn_meta.push_back(to_i32_checked(causal_start_s, "causal_start"));
            m_xattn_meta.push_back(to_i32_checked(q_start_strided_s, "q_start_strided"));
            m_xattn_meta.push_back(to_i32_checked(cumul_kq_max_bytes, "cumul_kq_max_bytes"));
            m_xattn_meta.push_back(to_i32_checked(cumul_exp_sum_bytes, "cumul_exp_sum_bytes"));
            m_xattn_meta.push_back(to_i32_checked(cumul_mask_elems, "cumul_mask_elems"));
            m_xattn_meta.push_back(to_i32_checked(cumul_mask_wg_elems, "cumul_mask_wg_elems"));
            m_xattn_meta.push_back(bi_begin);
            // Store global WG start offset for this subsequence (prefix sum before adding this subsequence's wg_count_s).
            m_xattn_meta.push_back(to_i32_checked(total_wg_count, "total_wg_count"));

            for (size_t m = 0; m < q_block_pad_s; ++m) {
                m_xattn_find_wg_map.push_back(subseq_id);
                m_xattn_find_wg_map.push_back(to_i32_checked(m, "find_q_block_idx"));
            }

            for (size_t m_merged = 0; m_merged < merged_q_blocks_s; ++m_merged) {
                m_xattn_post_wg_map.push_back(subseq_id);
                m_xattn_post_wg_map.push_back(to_i32_checked(m_merged, "post_merged_q_block_idx"));
            }

            cumul_kq_max_bytes += N_kq_groups_s * q_stride_pad_s * sizeof_softmax * heads_num;
            cumul_exp_sum_bytes += q_stride_pad_s * k_block_pad_s * sizeof_softmax * heads_num;
            cumul_mask_elems += q_block_pad_s * k_block_pad_s * heads_num;
            cumul_mask_wg_elems += merged_q_blocks_s * k_block_pad_s * heads_num;
            // Advance prefix sum so the next subsequence gets its own global WG start offset.
            total_wg_count += wg_count_s;

            max_q_block_pad = std::max(max_q_block_pad, q_block_pad_s);
            max_merged_q_blocks = std::max(max_merged_q_blocks, merged_q_blocks_s);
            num_xattn_subseqs++;
        }

        rt_params->xattn_block_size = block_size;
        rt_params->xattn_num_subseqs = num_xattn_subseqs;
        rt_params->xattn_gemmqk_wg_count = total_wg_count;
        rt_params->xattn_cumul_kq_max_bytes = cumul_kq_max_bytes;
        rt_params->xattn_cumul_exp_sum_bytes = cumul_exp_sum_bytes;
        rt_params->xattn_cumul_mask_elems = cumul_mask_elems;
        rt_params->xattn_cumul_mask_wg_elems = cumul_mask_wg_elems;
        rt_params->xattn_meta_num_int32s = m_xattn_meta.size();
        rt_params->xattn_find_wg_count = m_xattn_find_wg_map.size() / 2;
        rt_params->xattn_post_wg_count = m_xattn_post_wg_map.size() / 2;
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
        rt_params->batch_size_in_sequences = get_batch_size_in_sequences(params.input_layouts);
        rt_params->single_token_selected_count = 0;
        rt_params->multi_token_wg_count = 0;
        rt_params->enable_xattn_estimation = false;

        rt_params->stage = get_paged_attention_stage(params);
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
            const auto subsequence_begins = read_i32_input(params, PagedAttentionInputIdx::SUBSEQUENCE_BEGINS);
            const auto past_lens = read_i32_input(params, PagedAttentionInputIdx::PAST_LENS);
            const auto wg_seq_len = PagedAttentionGeneratorMultiToken::get_wg_seq_len(params);
            const bool use_split_mixed = rt_params->stage == PagedAttentionStage::MIXED && m_mixed_route_mode == MixedRouteMode::SPLIT;
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

                rt_params->enable_xattn_estimation = true;
                update_xattn_rt_params(instance);
            } else {
                rt_params->xattn_block_size = 1;  // disable xattn for pa
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

        const auto subsequence_begins = read_i32_input(params, PagedAttentionInputIdx::SUBSEQUENCE_BEGINS);

        const auto wg_seq_len = static_cast<int32_t>(PagedAttentionGeneratorMultiToken::get_wg_seq_len(params));
        auto mapping_mem = instance.get_intermediates_memories()[PagedAttentionInternBuffIdx::MULTI_TOKEN_WG_MAPPING];
        mem_lock<int32_t, mem_lock_type::write> mapping_lock(mapping_mem, stream);
        const size_t expected_mapping_size = rt_params->multi_token_wg_count * 2;
        OPENVINO_ASSERT(mapping_mem->count() >= expected_mapping_size,
                        "Insufficient multi-token mapping buffer size: expected at least ",
                        expected_mapping_size,
                        ", got ",
                        mapping_mem->count());
        size_t mapping_idx = 0;

        for (size_t subsequence_id = 0; subsequence_id + 1 < subsequence_begins.size(); ++subsequence_id) {
            const int32_t q_begin = subsequence_begins[subsequence_id];
            const int32_t q_end = subsequence_begins[subsequence_id + 1];

            if (q_end <= q_begin) {
                continue;
            }

            for (int32_t block_start = q_begin; block_start < q_end; block_start += wg_seq_len) {
                OPENVINO_ASSERT(mapping_idx + 1 < expected_mapping_size,
                                "Multi-token mapping write out of bounds: idx=",
                                mapping_idx,
                                ", expected_size=",
                                expected_mapping_size);
                mapping_lock[mapping_idx++] = block_start;
                mapping_lock[mapping_idx++] = static_cast<int32_t>(subsequence_id);
            }
        }

        OPENVINO_ASSERT(mapping_idx == expected_mapping_size, "Unexpected multi-token mapping size: expected ", expected_mapping_size, ", got ", mapping_idx);
    }

    void prepare_single_token_selected_ids(primitive_inst& instance) {
        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        OPENVINO_ASSERT(rt_params != nullptr);
        OPENVINO_ASSERT(rt_params->stage == PagedAttentionStage::GENERATE, "prepare_single_token_selected_ids is expected only for generate/decode stage");
        auto& stream = instance.get_network().get_stream();

        auto selected_ids_mem = instance.get_intermediates_memories()[PagedAttentionInternBuffIdx::SINGLE_TOKEN_SELECTED_SEQ_IDS];

        mem_lock<int32_t, mem_lock_type::write> selected_ids_lock(selected_ids_mem, stream);
        const size_t selected_capacity = selected_ids_mem->count();
        OPENVINO_ASSERT(selected_capacity >= rt_params->batch_size_in_sequences,
                        "Insufficient single-token selected ids buffer size: expected at least ",
                        rt_params->batch_size_in_sequences,
                        ", got ",
                        selected_capacity);
        std::iota(selected_ids_lock.begin(), selected_ids_lock.begin() + static_cast<std::ptrdiff_t>(rt_params->batch_size_in_sequences), 0);

        rt_params->single_token_selected_count = rt_params->batch_size_in_sequences;
    }

    void prepare_split_mixed_selected_ids_and_mapping(primitive_inst& instance) {
        const auto& params = *instance.get_impl_params();
        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        OPENVINO_ASSERT(rt_params != nullptr);
        OPENVINO_ASSERT(rt_params->stage == PagedAttentionStage::MIXED && m_mixed_route_mode == MixedRouteMode::SPLIT,
                        "prepare_split_mixed_selected_ids_and_mapping must be used only in split mixed mode");

        auto& stream = instance.get_network().get_stream();

        const auto subsequence_begins = read_i32_input(params, PagedAttentionInputIdx::SUBSEQUENCE_BEGINS);
        const auto past_lens = read_i32_input(params, PagedAttentionInputIdx::PAST_LENS);

        auto selected_ids_mem = instance.get_intermediates_memories()[PagedAttentionInternBuffIdx::SINGLE_TOKEN_SELECTED_SEQ_IDS];
        auto mapping_mem = instance.get_intermediates_memories()[PagedAttentionInternBuffIdx::MULTI_TOKEN_WG_MAPPING];
        mem_lock<int32_t, mem_lock_type::write> selected_ids_lock(selected_ids_mem, stream);
        mem_lock<int32_t, mem_lock_type::write> mapping_lock(mapping_mem, stream);

        const size_t selected_capacity = selected_ids_mem->count();
        const size_t expected_mapping_size = rt_params->multi_token_wg_count * 2;
        OPENVINO_ASSERT(mapping_mem->count() >= expected_mapping_size,
                        "Insufficient multi-token mapping buffer size: expected at least ",
                        expected_mapping_size,
                        ", got ",
                        mapping_mem->count());

        const int32_t wg_seq_len = static_cast<int32_t>(PagedAttentionGeneratorMultiToken::get_wg_seq_len(params));
        size_t selected_count = 0;
        size_t mapping_idx = 0;

        for (size_t sequence_id = 0; sequence_id + 1 < subsequence_begins.size(); ++sequence_id) {
            const int32_t q_begin = subsequence_begins[sequence_id];
            const int32_t q_end = subsequence_begins[sequence_id + 1];
            if (q_end <= q_begin) {
                continue;
            }

            const int32_t q_len = q_end - q_begin;
            const int32_t past_len = std::max<int32_t>(past_lens[sequence_id], 0);
            const bool decode_subsequence = (q_len == 1) && (past_len > 0);

            if (decode_subsequence) {
                OPENVINO_ASSERT(selected_count < selected_capacity,
                                "Single-token selected ids write out of bounds: idx=",
                                selected_count,
                                ", capacity=",
                                selected_capacity);
                selected_ids_lock[selected_count++] = static_cast<int32_t>(sequence_id);
                continue;
            }

            for (int32_t block_start = q_begin; block_start < q_end; block_start += wg_seq_len) {
                OPENVINO_ASSERT(mapping_idx + 1 < expected_mapping_size,
                                "Multi-token mapping write out of bounds: idx=",
                                mapping_idx,
                                ", expected_size=",
                                expected_mapping_size);
                mapping_lock[mapping_idx++] = block_start;
                mapping_lock[mapping_idx++] = static_cast<int32_t>(sequence_id);
            }
        }

        OPENVINO_ASSERT(selected_count == rt_params->single_token_selected_count,
                        "Unexpected selected ids count in split mixed mode: expected ",
                        rt_params->single_token_selected_count,
                        ", got ",
                        selected_count);
        OPENVINO_ASSERT(mapping_idx == expected_mapping_size, "Unexpected multi-token mapping size: expected ", expected_mapping_size, ", got ", mapping_idx);

        rt_params->single_token_selected_count = selected_count;
    }

    void prepare_xattn_metadata(primitive_inst& instance) {
        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        OPENVINO_ASSERT(rt_params != nullptr);
        if (!rt_params->enable_xattn_estimation || m_xattn_meta.empty())
            return;

        auto& stream = instance.get_network().get_stream();
        auto meta_mem = instance.get_intermediates_memories()[PagedAttentionInternBuffIdx::XATTN_SUBSEQ_META];
        OPENVINO_ASSERT(meta_mem->count() >= m_xattn_meta.size(),
                        "Insufficient xattn metadata buffer size: expected at least ",
                        m_xattn_meta.size(),
                        ", got ",
                        meta_mem->count());
        mem_lock<int32_t, mem_lock_type::write> meta_lock(meta_mem, stream);
        std::copy_n(m_xattn_meta.begin(), m_xattn_meta.size(), meta_lock.begin());

        auto find_map_mem = instance.get_intermediates_memories()[PagedAttentionInternBuffIdx::XATTN_FIND_WG_MAP];
        OPENVINO_ASSERT(find_map_mem->count() >= m_xattn_find_wg_map.size(),
                        "Insufficient xattn find-map buffer size: expected at least ",
                        m_xattn_find_wg_map.size(),
                        ", got ",
                        find_map_mem->count());
        mem_lock<int32_t, mem_lock_type::write> find_map_lock(find_map_mem, stream);
        std::copy_n(m_xattn_find_wg_map.begin(), m_xattn_find_wg_map.size(), find_map_lock.begin());

        auto post_map_mem = instance.get_intermediates_memories()[PagedAttentionInternBuffIdx::XATTN_POST_WG_MAP];
        OPENVINO_ASSERT(post_map_mem->count() >= m_xattn_post_wg_map.size(),
                        "Insufficient xattn post-map buffer size: expected at least ",
                        m_xattn_post_wg_map.size(),
                        ", got ",
                        post_map_mem->count());
        mem_lock<int32_t, mem_lock_type::write> post_map_lock(post_map_mem, stream);
        std::copy_n(m_xattn_post_wg_map.begin(), m_xattn_post_wg_map.size(), post_map_lock.begin());
    }

    // update impl_parameter and rt_parameter
    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        PrimitiveImplCM::update(inst, impl_params);
        update_rt_params(inst);
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        const auto& params = *instance.get_impl_params();
        const auto desc = params.typed_desc<paged_attention>();

        update_stages_flags(instance);
        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        OPENVINO_ASSERT(rt_params != nullptr);

        GPU_DEBUG_TRACE_DETAIL << "ov::intel_gpu::cm::PagedAttentionCmImpl::execute():  stage = " << static_cast<int>(rt_params->stage) << std::endl;
        std::vector<event::ptr> res_event = events;
        res_event = {execute_stage(res_event, instance, kv_cache_update)};

        const auto execute_multi_token_path = [&]() {
            if (rt_params->multi_token_wg_count == 0) {
                return;
            }

            const bool xattn_enabled = desc->has_xattention && rt_params->enable_xattn_estimation;
            const bool xattn_disabled = !xattn_enabled || bypass_xattn(params);
            if (xattn_disabled) {
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

                prepare_xattn_metadata(instance);
                res_event = {execute_stage(res_event, instance, xattn_gemmqk)};
                XATTN_DUMP(instance, PagedAttentionInternBuffIdx::XATTN_GEMMQK_MAX);  // 2: kq_max_wg
                XATTN_DUMP(instance,
                           PagedAttentionInternBuffIdx::XATTN_GEMMQK_EXPSUMS);  // idx 3: kq_exp_partial_sum is subject to change in find_block kernel.
                res_event = {execute_stage(res_event, instance, xattn_find_block)};
                XATTN_DUMP(instance, PagedAttentionInternBuffIdx::XATTN_BLOCKMASK);  // 4: sparse_block_mask
#if FIND_DEBUG_ACC
                XATTN_DUMP(instance, PagedAttentionInternBuffIdx::XATTN_FIND_DEBUG_ACC);
#endif
                res_event = {execute_stage(res_event, instance, xattn_post_proc)};
                res_event = {execute_stage(res_event, instance, pa_multi_token)};
            }
        };

        if (rt_params->stage == PagedAttentionStage::PREFILL) {
            prepare_multi_token_mapping(instance);
            execute_multi_token_path();
        } else if (rt_params->stage == PagedAttentionStage::MIXED) {
            GPU_DEBUG_TRACE_DETAIL << "Execute Mixed stage with mode: " << (m_mixed_route_mode == MixedRouteMode::SPLIT ? "SPLIT" : "MERGE") << ", "
                                   << rt_params->single_token_selected_count << " single token selected and " << rt_params->multi_token_wg_count
                                   << " workgroups for multi token." << std::endl;
            if (m_mixed_route_mode == MixedRouteMode::SPLIT) {
                prepare_split_mixed_selected_ids_and_mapping(instance);
                if (rt_params->single_token_selected_count > 0) {
                    res_event = {execute_stage(res_event, instance, pa_single_token)};
                    res_event = {execute_stage(res_event, instance, pa_single_token_finalization)};
                }
                execute_multi_token_path();
            } else {
                prepare_multi_token_mapping(instance);
                execute_multi_token_path();
            }
        } else {
            prepare_single_token_selected_ids(instance);
            res_event = {execute_stage(res_event, instance, pa_single_token)};
            res_event = {execute_stage(res_event, instance, pa_single_token_finalization)};
        }

        return res_event[0];
    }

    bool requires_update(primitive_inst& inst, const kernel_impl_params& impl_params) const override {
        const auto stage = get_paged_attention_stage(impl_params);

        // In case of MIXED mode execution Paged Attention may require dispatch data update and internal
        // buffers reallocation even if the input shapes haven't been changed. Therefore, check the current execution
        // mode and update parameters if needed
        return stage == PagedAttentionStage::MIXED ||
               ((stage == PagedAttentionStage::PREFILL || stage == PagedAttentionStage::UNKNOWN) && get_batch_size_in_sequences(impl_params.input_layouts) > 1);
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
            internal_buffers.emplace_back(2, ov::element::i32, true, false);          // 2: unused multi-token mapping placeholder (lockable, not shareable)
            internal_buffers.emplace_back(total_tokens,
                                          ov::element::i32,
                                          true,
                                          false);  // 3: selected sequence ids (lockable for mem_lock<write>, not shareable)

            GPU_DEBUG_TRACE_DETAIL << "  internal buffer sizes: tmp_out=" << tmp_out_elements_count * 4 << "  exp_sums=" << buf_elements_count * 4 << std::endl;
        } else {
            int64_t decode_tmp_out_elements_count = 16;
            int64_t decode_buf_elements_count = 16;
            const bool needs_single_token_buffers =
                stage == PagedAttentionStage::MIXED && m_mixed_route_mode == MixedRouteMode::SPLIT && rt_params->single_token_selected_count > 0;
            if (needs_single_token_buffers) {
                OPENVINO_ASSERT(rt_params->num_of_partitions != 0);
                decode_buf_elements_count = static_cast<int64_t>(rt_params->single_token_selected_count * desc->heads_num * rt_params->num_of_partitions);
                decode_tmp_out_elements_count =
                    static_cast<int64_t>(rt_params->single_token_selected_count * desc->heads_num * desc->v_head_size * rt_params->num_of_partitions);
            }

            internal_buffers.emplace_back(decode_tmp_out_elements_count, ov::element::f32);  // 0: intermediate partition output
            internal_buffers.emplace_back(decode_buf_elements_count, ov::element::f32);      // 1: softmax exp_sums
            internal_buffers.emplace_back(std::max<int64_t>(2, static_cast<int64_t>(rt_params->multi_token_wg_count * 2)),
                                          ov::element::i32,
                                          true,
                                          false);  // 2: multi-token mapping (lockable for mem_lock<write>, not shareable)
            internal_buffers.emplace_back(std::max<int64_t>(1, static_cast<int64_t>(rt_params->batch_size_in_sequences)),
                                          ov::element::i32,
                                          true,
                                          false);  // 3: selected ids (lockable for mem_lock<write>, not shareable)

            // internal buffer for XAttention (cumulative sizes across all subsequences)
            if (rt_params->enable_xattn_estimation) {
                auto count_kq_max_wg = static_cast<int64_t>(rt_params->xattn_cumul_kq_max_bytes / sizeof(float));
                internal_buffers.emplace_back(std::max<int64_t>(1, count_kq_max_wg), ov::element::f32);  // 4: kq_max_wg

                auto count_kq_exp_partial_sum = static_cast<int64_t>(rt_params->xattn_cumul_exp_sum_bytes / sizeof(float));
                internal_buffers.emplace_back(std::max<int64_t>(1, count_kq_exp_partial_sum), ov::element::f32);  // 5: kq_exp_partial_sum

                auto count_elements_mask = static_cast<int64_t>(rt_params->xattn_cumul_mask_elems);
                internal_buffers.emplace_back(std::max<int64_t>(1, count_elements_mask), ov::element::boolean);  // 6: sparse_block_mask

                auto count_elements_mask_merged = static_cast<int64_t>(rt_params->xattn_cumul_mask_wg_elems);
                internal_buffers.emplace_back(std::max<int64_t>(1, count_elements_mask_merged), ov::element::boolean);  // 7: sparse_block_mask_wg

                auto count_meta_int32s = static_cast<int64_t>(rt_params->xattn_meta_num_int32s);
                internal_buffers.emplace_back(std::max<int64_t>(16, count_meta_int32s),
                                              ov::element::i32,
                                              true,
                                              false);  // 8: xattn_subseq_meta (lockable for mem_lock<write>, not shareable)

                auto count_find_map_int32s = static_cast<int64_t>(rt_params->xattn_find_wg_count * 2);
                internal_buffers.emplace_back(std::max<int64_t>(2, count_find_map_int32s),
                                              ov::element::i32,
                                              true,
                                              false);  // 9: xattn_find_wg_map (lockable for mem_lock<write>, not shareable)

                auto count_post_map_int32s = static_cast<int64_t>(rt_params->xattn_post_wg_count * 2);
                internal_buffers.emplace_back(std::max<int64_t>(2, count_post_map_int32s),
                                              ov::element::i32,
                                              true,
                                              false);  // 10: xattn_post_wg_map (lockable for mem_lock<write>, not shareable)

#if FIND_DEBUG_ACC
                auto count_elements_kq_sum = static_cast<int64_t>(rt_params->xattn_cumul_mask_elems);
                internal_buffers.emplace_back(std::max<int64_t>(1, count_elements_kq_sum), ov::element::f16);  // 11: kq_sum
#endif

                GPU_DEBUG_TRACE_DETAIL << "  internal buffer sizes: count_kq_max_wg=" << count_kq_max_wg * 4
                                       << "  count_kq_exp_partial_sum=" << count_kq_exp_partial_sum * 4 << "  count_elements_mask=" << count_elements_mask
                                       << "  count_elements_mask_merged=" << count_elements_mask_merged << "  count_meta_int32s=" << count_meta_int32s
                                       << "  count_find_map_int32s=" << count_find_map_int32s << "  count_post_map_int32s=" << count_post_map_int32s
                                       << std::endl;
            }
        }

        return internal_buffers;
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        auto copy = make_deep_copy<PagedAttentionCmImpl>(this);
        copy->m_mixed_route_mode = m_mixed_route_mode;
        return copy;
    }

private:
    std::vector<int32_t> m_xattn_meta;
    std::vector<int32_t> m_xattn_find_wg_map;
    std::vector<int32_t> m_xattn_post_wg_map;
    MixedRouteMode m_mixed_route_mode = MixedRouteMode::SPLIT;

    void validate_xattn_inputs(const kernel_impl_params& params, size_t batch_size) {
        const auto& input_mem = params.memory_deps;

        auto validate_input = [&](size_t idx, const char* name) {
            const auto it = input_mem.find(idx);
            if (it == input_mem.end() || it->second == nullptr)
                OPENVINO_THROW("XAttention ", name, " input is required at index ", idx);

            const auto input_size = it->second->count();
            if (input_size != 1 && input_size != batch_size)
                OPENVINO_THROW("XAttention ", name, " input size (", input_size, ") must be 1 or equal to batch size (", batch_size, ")");
        };

        validate_input(PagedAttentionInputIdx::XATTENTION_BLOCK_SIZE, "block size");
        validate_input(PagedAttentionInputIdx::XATTENTION_THRESHOLD, "threshold");
    }

    size_t get_xattn_block_size(const kernel_impl_params& params, const size_t seq_idx = 0) {
        constexpr int32_t block_size_128 = 128;
        constexpr int32_t block_size_256 = 256;

        const auto rt_params = static_cast<const PagedAttentionRuntimeParams*>(m_rt_params.get());
        OPENVINO_ASSERT(rt_params != nullptr, "PagedAttention runtime params are not initialized");
        OPENVINO_ASSERT(rt_params->enable_xattn_estimation, "XAttention block size must be accessed only when enable_xattn_estimation is true");

        const auto desc = params.typed_desc<paged_attention>();
        OPENVINO_ASSERT(desc->has_xattention, "XAttention block size must be accessed only when has_xattention is true");

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
