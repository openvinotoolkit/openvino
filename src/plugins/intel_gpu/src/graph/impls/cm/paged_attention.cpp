// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention.hpp"
#include "paged_attention_gen.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

#include "primitive_cm_base.hpp"
#include "common_utils/kernel_generator_base.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "kv_cache_inst.h"
#include "openvino/core/partial_shape.hpp"
#include "paged_attention_inst.h"
#include "primitive_inst.h"

#define DUMP_XATTN_BLOCK_MASK 0
#if DUMP_XATTN_BLOCK_MASK
#include "openvino/util/file_util.hpp"
#endif

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

    PagedAttentionCmImpl(): PrimitiveImplCM(PagedAttentionImplementationManager::get_type_info_static()) {
        m_rt_params = std::make_unique<PagedAttentionRuntimeParams>();
    }
    explicit PagedAttentionCmImpl(const kernel_impl_params& params) : PagedAttentionCmImpl() {
        const auto desc = params.typed_desc<paged_attention>();

        GPU_DEBUG_TRACE_DETAIL << "ov::intel_gpu::cm::PagedAttentionCmImpl::PagedAttentionCmImpl()" << std::endl;
        add_stage(kv_cache_update, params);
        add_stage(pa_single_token, params);
        add_stage(pa_single_token_finalization, params);
        add_stage(pa_multi_token, params);
        const size_t xattn_block_size = get_xattn_block_size(params);
        if (xattn_block_size > 1) {
            add_stage(xattn_estimate_gemmqk, params);
            add_stage(xattn_estimate_find_block, params);
        }
    }

    void update_rt_params(const primitive_inst& instance) override {
        update_stages_flags(instance);
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<PagedAttentionRuntimeParams>();
        }
        GPU_DEBUG_TRACE_DETAIL << "ov::intel_gpu::cm::PagedAttentionCmImpl::update_rt_params()" << std::endl;
        const auto& params = *instance.get_impl_params();
        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        const auto& desc = params.typed_desc<paged_attention>();

        const auto max_context_len = get_max_context_len(params);
        rt_params->max_context_len = max_context_len;
        rt_params->partition_size = get_partition_size();
        rt_params->num_of_partitions = ceil_div(max_context_len, rt_params->partition_size);
        rt_params->stage = get_paged_attention_stage(params);

        GPU_DEBUG_TRACE_DETAIL << "  max_context_len: " << rt_params->max_context_len << "  partition_size: " << rt_params->partition_size
                               << "  num_of_partitions: " << rt_params->num_of_partitions << ", stage: " << static_cast<size_t>(rt_params->stage) << std::endl;
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
        assert(rt_params != nullptr);

        GPU_DEBUG_TRACE_DETAIL << "ov::intel_gpu::cm::PagedAttentionCmImpl::execute():  stage = " << static_cast<int>(rt_params->stage) << std::endl;
        std::vector<event::ptr> res_event = events;
        res_event = {execute_stage(res_event, instance, kv_cache_update)};

        if (rt_params->stage == PagedAttentionStage::PREFILL || rt_params->stage == PagedAttentionStage::MIXED) {
            if (has_stage(xattn_estimate_gemmqk)) {
                // cldnn::stream& stream = instance.get_network().get_stream();
                // stream.finish();
                res_event = {execute_stage(res_event, instance, xattn_estimate_gemmqk)};
                // stream.finish();
                // std::cout << "finish xattn_estimate_gemmqk!\n";
                res_event = {execute_stage(res_event, instance, xattn_estimate_find_block)};
#if DUMP_XATTN_BLOCK_MASK
                {
                    cldnn::stream& stream = instance.get_network().get_stream();
                    stream.finish();
                    static uint32_t pa_id = 0;
                    std::cout << "finish xattn_estimate_find_block!\n";
                    auto output_mem = instance.get_intermediates_memories()[4];
                    mem_lock<char, mem_lock_type::read> lock(output_mem, stream);
                    auto& layout = output_mem->get_layout();
                    std::string data_type = ov::element::Type(layout.data_type).get_type_name();
                    std::string format = layout.format.to_string();
                    std::string tensor;
                    auto dims = layout.get_dims();
                    for (size_t r = 0 ; r < layout.get_rank() ; r++) {
                        tensor += ("_" + to_string(dims[r]));
                    }
                    // std::string filename = "PA" + std::to_string(pa_id) + "__" + data_type + "_" + tensor + "__" + format + ".bin";
                    std::string filename = "PA" + std::to_string(pa_id) + ".bin";
                    ov::util::save_binary(filename, lock.data(), output_mem->size());
                    pa_id++;
                }
#endif
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
        const auto indexes_dt = ov::element::f32;
        auto stage = PagedAttentionStage::UNKNOWN;
        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());

        size_t partition_size = PA_KV_CACHE_BLOCK_SIZE;
        size_t num_of_partitions = 1;
        if (rt_params != nullptr && rt_params->num_of_partitions != 0) {
            stage = rt_params->stage;
            partition_size = rt_params->partition_size;
            num_of_partitions = rt_params->num_of_partitions;
        } else {
            stage = get_paged_attention_stage(params);
            const auto max_context_len = get_max_context_len(params);
            partition_size = get_partition_size();
            num_of_partitions = ceil_div(max_context_len, partition_size);
        }
        GPU_DEBUG_TRACE_DETAIL << "ov::intel_gpu::cm::PagedAttentionCmImpl::get_internal_buffer_descs(): stage = " << static_cast<int>(stage)
                               << "  partition_size: " << partition_size << "  num_of_partitions: " << num_of_partitions << std::endl;
        if (stage == PagedAttentionStage::GENERATE) {
            const auto& input = params.input_layouts[0];
            const int64_t total_tokens = input.get_partial_shape()[0].get_length();
            auto buf_elements_count = static_cast<int64_t>(total_tokens * desc->heads_num * num_of_partitions);
            auto tmp_out_elements_count = static_cast<int64_t>(total_tokens * desc->heads_num * desc->v_head_size * num_of_partitions);

            internal_buffers.emplace_back(tmp_out_elements_count, ov::element::f32);  // 0: intermediate partition output
            internal_buffers.emplace_back(buf_elements_count, ov::element::f32);      // 1: softmax exp_sums

            GPU_DEBUG_TRACE_DETAIL << "  internal buffer sizes: tmp_out=" << tmp_out_elements_count * 4 << "  exp_sums=" << buf_elements_count * 4 << std::endl;
        } else {
            internal_buffers.emplace_back(16, indexes_dt);  // 0: intermediate partition output
            internal_buffers.emplace_back(16, indexes_dt);  // 1: softmax exp_sums

            // internal buffer for XAttention
            auto out_shape = params.output_layouts[0].get_shape();
            const size_t kv_len = get_max_context_len(params) / STRIDE * STRIDE;
            const size_t q_len = out_shape[0];
            const uint32_t M = static_cast<uint32_t>(q_len / STRIDE);   //# will slient drop the tails which is less than `stride`
            const uint32_t N = static_cast<uint32_t>(kv_len / STRIDE);
            const size_t q_stride_pad = round_up_to(M, BLOCK_WG_M);
            const size_t N_kq_groups = ceil_div(N, BLOCK_WG_N);

            auto count_kq_max_wg = static_cast<int64_t>(desc->heads_num * N_kq_groups * q_stride_pad);
            internal_buffers.emplace_back(count_kq_max_wg, ov::element::f32);                // 2: kq_max_wg

            const size_t block_size = get_xattn_block_size(params);
            if (block_size > 1) {
                OPENVINO_ASSERT(block_size % STRIDE == 0, "sparse block_size must be devidable by stride.");
                const uint32_t q_block_pad = ceil_div(q_len, block_size);
                const uint32_t sum_per_token_in_block = static_cast<uint32_t>(block_size / STRIDE);
                const uint32_t k_block_in_group = static_cast<uint32_t>(BLOCK_WG_N / sum_per_token_in_block);
                const uint32_t k_block_pad = k_block_in_group * N_kq_groups;
                auto count_kq_exp_partial_sum = static_cast<int64_t>(desc->heads_num * q_stride_pad * k_block_pad);
                internal_buffers.emplace_back(count_kq_exp_partial_sum, ov::element::f32);       // 3: kq_exp_partial_sum

                auto count_elements_mask = static_cast<int64_t>(desc->heads_num * q_block_pad * k_block_pad);
                internal_buffers.emplace_back(count_elements_mask, ov::element::boolean);        // 4: sparse_block_mask
            }
        }

        return internal_buffers;
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<PagedAttentionCmImpl>(this);
    }
};

std::unique_ptr<primitive_impl> PagedAttentionImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<paged_attention>());
    try {
        return std::make_unique<PagedAttentionCmImpl>(params);
    } catch (const std::exception& e) {
        OPENVINO_THROW("Failed to create PagedAttentionCmImpl: ", e.what());
    }
}

} // namespace ov::intel_gpu::cm
// BIND_BINARY_BUFFER_WITH_TYPE(cldnn::paged_attention)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::PagedAttentionCmImpl)