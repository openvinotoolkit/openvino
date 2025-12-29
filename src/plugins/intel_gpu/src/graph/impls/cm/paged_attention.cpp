// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention.hpp"

#include <array>
#include <cstdint>
#include <memory>
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

namespace ov::intel_gpu::cm {

class PagedAttentionCmImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::PagedAttentionCmImpl)

    Stage::Ptr kv_cache_update = make_stage<PagedAttentionGeneratorKVCacheUpdate>();
    Stage::Ptr pa_single_token = make_stage<PagedAttentionGeneratorSingleToken>();
    Stage::Ptr pa_single_token_finalization = make_stage<PagedAttentionGeneratorSingleTokenFinalization>();
    Stage::Ptr pa_multi_token = make_stage<PagedAttentionGeneratorMultiToken>();
    Stage::Ptr xattn_estimate_gemmqk = make_stage<XAttentionEstimateGEMMQK>();
    Stage::Ptr xattn_estimate_find_block = make_stage<XAttentionEstimateFindBlock>();
    Stage::Ptr xattn_estimate_post_proc = make_stage<XAttentionEstimatePostProc>();

    PagedAttentionCmImpl() : PrimitiveImplCM(PagedAttentionImplementationManager::get_type_info_static()) {
        m_rt_params = std::make_unique<PagedAttentionRuntimeParams>();
    }
    explicit PagedAttentionCmImpl(const kernel_impl_params& params) : PagedAttentionCmImpl() {
        const auto desc = params.typed_desc<paged_attention>();

        GPU_DEBUG_TRACE_DETAIL << "ov::intel_gpu::cm::PagedAttentionCmImpl::PagedAttentionCmImpl()" << std::endl;
        add_stage(kv_cache_update, params);
        add_stage(pa_single_token, params);
        add_stage(pa_single_token_finalization, params);
        add_stage(pa_multi_token, params);
        if (desc->has_xattention) {
            add_stage(xattn_estimate_gemmqk, params);
            add_stage(xattn_estimate_find_block, params);
            add_stage(xattn_estimate_post_proc, params);
        }
    }

    void update_xattn_rt_params(const kernel_impl_params& params) {
        const auto desc = params.typed_desc<paged_attention>();

        // XAttention estimate is following afer kvcache_update.
        auto out_shape = params.output_layouts[0].get_shape();
        const size_t block_size = get_xattn_block_size(params);
        const size_t kv_len = get_max_context_len(params);
        const size_t q_len = out_shape[0];
        const size_t N = kv_len / STRIDE;
        const size_t N_kq_groups = ceil_div(N, BLOCK_WG_N);

        const auto q_block_pad = ceil_div(q_len, block_size);
        const auto sum_per_token_in_block = block_size / STRIDE;
        const auto k_block_in_group = BLOCK_WG_N / sum_per_token_in_block;
        const auto k_block_pad = k_block_in_group * N_kq_groups;

        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        rt_params->q_block_pad = q_block_pad;
        rt_params->k_block_pad = k_block_pad;
        rt_params->q_block_pad_merged = ceil_div(q_block_pad, MERGED_Q_NUM);

        const size_t head_size = desc->k_head_size;

        const auto M = q_len / STRIDE;  //# will slient drop the tails which is less than `stride`
        const auto K = STRIDE * head_size;

        const size_t q_stride_pad = round_up_to(M, BLOCK_WG_M);

        rt_params->N_kq_groups = N_kq_groups;
        rt_params->M = M;
        rt_params->N = N;
        rt_params->K = K;
        rt_params->q_stride_pad = q_stride_pad;
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

        rt_params->stage = get_paged_attention_stage(params);
        const auto max_context_len = get_max_context_len(params);
        rt_params->max_context_len = max_context_len;
        GPU_DEBUG_TRACE_DETAIL << "update_rt_params for stage: " << static_cast<size_t>(rt_params->stage) << "  max_context_len: " << rt_params->max_context_len
                               << std::endl;

        if (rt_params->stage == PagedAttentionStage::GENERATE) {
            auto partition_size = get_partition_size(desc->has_xattention);
            rt_params->num_of_partitions = ceil_div(max_context_len, partition_size);

            GPU_DEBUG_TRACE_DETAIL << "  partition_size: " << partition_size << "  num_of_partitions: " << rt_params->num_of_partitions << std::endl;
        } else {
            if (desc->has_xattention) {
                update_xattn_rt_params(params);
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

        update_stages_flags(instance);
        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        OPENVINO_ASSERT(rt_params != nullptr);

        GPU_DEBUG_TRACE_DETAIL << "ov::intel_gpu::cm::PagedAttentionCmImpl::execute():  stage = " << static_cast<int>(rt_params->stage) << std::endl;
        std::vector<event::ptr> res_event = events;
        res_event = {execute_stage(res_event, instance, kv_cache_update)};

        if (rt_params->stage == PagedAttentionStage::PREFILL || rt_params->stage == PagedAttentionStage::MIXED) {
            if (has_stage(xattn_estimate_gemmqk) && !bypass_xattn(params)) {
                res_event = {execute_stage(res_event, instance, xattn_estimate_gemmqk)};
                res_event = {execute_stage(res_event, instance, xattn_estimate_find_block)};
                res_event = {execute_stage(res_event, instance, xattn_estimate_post_proc)};
            }
            res_event = {execute_stage(res_event, instance, pa_multi_token)};
        } else if (rt_params->stage == PagedAttentionStage::GENERATE) {
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
