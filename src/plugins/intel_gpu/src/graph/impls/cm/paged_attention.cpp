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

namespace ov::intel_gpu::cm {

class PagedAttentionCmImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::PagedAttentionCmImpl)

    Stage::Ptr kv_cache_update = make_stage<PagedAttentionGeneratorKVCacheUpdate>();
    Stage::Ptr pa_single_token = make_stage<PagedAttentionGeneratorSingleToken>();
    Stage::Ptr pa_single_token_finalization = make_stage<PagedAttentionGeneratorSingleTokenFinalization>();
    Stage::Ptr pa_multi_token = make_stage<PagedAttentionGeneratorMultiToken>();

    PagedAttentionCmImpl(): PrimitiveImplCM(PagedAttentionImplementationManager::get_type_info_static()) {}
    explicit PagedAttentionCmImpl(const kernel_impl_params& params) : PagedAttentionCmImpl() {
        const auto desc = params.typed_desc<paged_attention>();

        std::cout << "ov::intel_gpu::cm::PagedAttentionCmImpl::PagedAttentionCmImpl()" << std::endl;
        add_stage(kv_cache_update, params);
        add_stage(pa_single_token, params);
        add_stage(pa_single_token_finalization, params);
        add_stage(pa_multi_token, params);
    }

    void update_rt_params(const primitive_inst& instance) override {
        update_stages_flags(instance);
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<ImplRuntimeParams>();
        }
        std::cout << "ov::intel_gpu::cm::PagedAttentionCmImpl::update_rt_params()" << std::endl;
        const auto& params = *instance.get_impl_params();
        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        const auto& desc = params.typed_desc<paged_attention>();

        const auto max_context_len = get_max_context_len(params);
        rt_params->max_context_len = max_context_len;
        rt_params->partition_size = get_partition_size();
        rt_params->num_of_partitions = ceil_div(max_context_len, rt_params->partition_size);
        rt_params->stage = get_paged_attention_stage(params);

        std::cout << "  max_context_len: " << rt_params->max_context_len << "  partition_size: " << rt_params->partition_size
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

        std::cout << "ov::intel_gpu::cm::PagedAttentionCmImpl::execute():  stage = " << static_cast<int>(rt_params->stage) << std::endl;
        std::vector<event::ptr> res_event = events;
        res_event = {execute_stage(res_event, instance, kv_cache_update)};

        if (rt_params->stage == PagedAttentionStage::PREFILL || rt_params->stage == PagedAttentionStage::MIXED) {
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
        const auto indexes_dt = ov::element::u8;
        const auto element_size = 4;  // 4 bytes
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
        std::cout << "ov::intel_gpu::cm::PagedAttentionCmImpl::get_internal_buffer_descs(): stage = " << static_cast<int>(stage)
                  << "  partition_size: " << partition_size << "  num_of_partitions: " << num_of_partitions << std::endl;
        if (stage == PagedAttentionStage::GENERATE) {
            const auto& input = params.input_layouts[0];
            const int64_t total_tokens = input.get_partial_shape()[0].get_length();
            auto buf_elements_count = static_cast<int64_t>(total_tokens * desc->heads_num * num_of_partitions);
            auto tmp_out_elements_count = static_cast<int64_t>(total_tokens * desc->heads_num * desc->v_head_size * num_of_partitions);

            internal_buffers.emplace_back(tmp_out_elements_count * element_size, indexes_dt);  // 0: intermediate partition output
            internal_buffers.emplace_back(buf_elements_count * element_size, indexes_dt);      // 1: softmax exp_sums

            std::cout << "  internal buffer sizes: tmp_out=" << tmp_out_elements_count * element_size << "  exp_sums=" << buf_elements_count * element_size
                      << std::endl;
        } else {
            internal_buffers.emplace_back(16, indexes_dt);  // 0: intermediate partition output
            internal_buffers.emplace_back(16, indexes_dt);  // 1: softmax exp_sums
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