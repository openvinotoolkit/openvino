// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_selective_ssm_executor.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>

#include "memory_desc/cpu_blocked_memory_desc.h"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/kernels/selective_ssm.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {
namespace {

bool is_supported_data_precision(const ov::element::Type& precision) {
    return precision == ov::element::f32 || precision == ov::element::f16 || precision == ov::element::bf16;
}

bool is_supported_index_precision(const ov::element::Type& precision) {
    return precision == ov::element::i32 || precision == ov::element::i64;
}

}  // namespace

bool PagedSelectiveSSMExecutor::supports(const PagedSelectiveSSMConfig& config) {
    const auto data_precision = config.descs.at(ARG_PAGED_SSM_A)->getPrecision();
    if (!is_supported_data_precision(data_precision)) {
        return false;
    }
    for (const auto arg : {ARG_PAGED_SSM_DT,
                           ARG_PAGED_SSM_B,
                           ARG_PAGED_SSM_X,
                           ARG_PAGED_SSM_C,
                           ARG_PAGED_SSM_STATE,
                           ARG_PAGED_SSM_OUT}) {
        if (config.descs.at(arg)->getPrecision() != data_precision) {
            return false;
        }
    }

    const auto index_precision = config.descs.at(ARG_PAGED_SSM_SUBSEQUENCE_BEGINS)->getPrecision();
    if (!is_supported_index_precision(index_precision)) {
        return false;
    }
    for (const auto arg : {ARG_PAGED_SSM_BLOCK_INDICES,
                           ARG_PAGED_SSM_BLOCK_INDICES_BEGINS,
                           ARG_PAGED_SSM_NUM_PROCESSED_TOKENS,
                           ARG_PAGED_SSM_CACHE_INTERVAL}) {
        if (config.descs.at(arg)->getPrecision() != index_precision) {
            return false;
        }
    }
    return true;
}

PagedSelectiveSSMExecutor::PagedSelectiveSSMExecutor(const PagedSelectiveSSMAttrs&,
                                                     const MemoryArgs& memory,
                                                     ExecutorContext::CPtr context)
    : m_context(std::move(context)) {
    update(memory);
}

bool PagedSelectiveSSMExecutor::update_scratchpad(const MemoryArgs& memory) {
    const auto& x_shape = memory.at(ARG_PAGED_SSM_X)->getDescPtr()->getShape();
    const auto& state_shape = memory.at(ARG_PAGED_SSM_STATE)->getDescPtr()->getShape();
    if (x_shape.isDynamic() || state_shape.isDynamic()) {
        return true;
    }
    const auto& x_dims = x_shape.getStaticDims();
    const auto& state_dims = state_shape.getStaticDims();
    OPENVINO_ASSERT(x_dims.size() == 3 && state_dims.size() == 4);
    const auto head_dim = x_dims[2];
    const auto state_size = state_dims[3];
    const auto physical_blocks = state_dims[0];
    OPENVINO_ASSERT(state_size > 0, "PagedSelectiveSSM state_size must be greater than zero.");
    const auto thread_count = static_cast<size_t>(m_context->getCpuParallel()->get_num_worker_threads());
    const auto scratch_head_dim = node::kernel::get_scratch_head_dim(head_dim, state_size, x_dims[1], thread_count);
    if (m_state_scratch && m_block_owners && m_scratch_head_dim == scratch_head_dim &&
        m_scratch_state_size == state_size && m_cached_physical_blocks == physical_blocks) {
        return true;
    }

    const auto state_desc =
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::f32,
                                               ov::intel_cpu::Shape{thread_count, scratch_head_dim * state_size});
    const auto owner_desc =
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::i32,
                                               ov::intel_cpu::Shape{std::max(size_t{1}, physical_blocks)});
    m_state_scratch = m_context->getScratchPad()->createScratchPadMem(state_desc);
    m_block_owners = m_context->getScratchPad()->createScratchPadMem(owner_desc);
    m_scratch_head_dim = scratch_head_dim;
    m_scratch_state_size = state_size;
    m_cached_physical_blocks = physical_blocks;
    return m_state_scratch != nullptr && m_block_owners != nullptr;
}

bool PagedSelectiveSSMExecutor::update(const MemoryArgs& memory) {
    return update_scratchpad(memory);
}

void PagedSelectiveSSMExecutor::execute(const MemoryArgs& memory) {
    const auto& B_dims = memory.at(ARG_PAGED_SSM_B)->getStaticDims();
    const auto& x_dims = memory.at(ARG_PAGED_SSM_X)->getStaticDims();
    const auto& state_dims = memory.at(ARG_PAGED_SSM_STATE)->getStaticDims();
    const auto& subsequence_dims = memory.at(ARG_PAGED_SSM_SUBSEQUENCE_BEGINS)->getStaticDims();
    const auto& block_indices_dims = memory.at(ARG_PAGED_SSM_BLOCK_INDICES)->getStaticDims();
    const auto& block_begins_dims = memory.at(ARG_PAGED_SSM_BLOCK_INDICES_BEGINS)->getStaticDims();
    const auto& processed_dims = memory.at(ARG_PAGED_SSM_NUM_PROCESSED_TOKENS)->getStaticDims();
    const auto& interval_dims = memory.at(ARG_PAGED_SSM_CACHE_INTERVAL)->getStaticDims();
    OPENVINO_ASSERT(B_dims.size() == 3 && x_dims.size() == 3 && state_dims.size() == 4);
    OPENVINO_ASSERT(subsequence_dims.size() == 1 && subsequence_dims[0] >= 1 && block_indices_dims.size() == 1);
    const auto sequence_count = subsequence_dims[0] - 1;
    OPENVINO_ASSERT(block_begins_dims.size() == 1 && processed_dims.size() == 1 && interval_dims.size() == 1 &&
                        block_begins_dims[0] == sequence_count + 1 && processed_dims[0] == sequence_count &&
                        interval_dims[0] == sequence_count,
                    "PagedSelectiveSSM metadata tensor lengths are inconsistent.");
    if (!m_state_scratch || !m_block_owners || m_scratch_state_size != state_dims[3] ||
        m_cached_physical_blocks != state_dims[0]) {
        OPENVINO_ASSERT(update_scratchpad(memory));
    }

    const node::kernel::PagedSelectiveSSMShape shape{x_dims[0],
                                                     x_dims[1],
                                                     x_dims[2],
                                                     B_dims[1],
                                                     B_dims[2],
                                                     state_dims[0],
                                                     block_indices_dims[0],
                                                     sequence_count};
    node::kernel::paged_selective_ssm(memory.at(ARG_PAGED_SSM_A)->getData(),
                                      memory.at(ARG_PAGED_SSM_DT)->getData(),
                                      memory.at(ARG_PAGED_SSM_B)->getData(),
                                      memory.at(ARG_PAGED_SSM_X)->getData(),
                                      memory.at(ARG_PAGED_SSM_C)->getData(),
                                      memory.at(ARG_PAGED_SSM_STATE)->getData(),
                                      memory.at(ARG_PAGED_SSM_SUBSEQUENCE_BEGINS)->getData(),
                                      memory.at(ARG_PAGED_SSM_BLOCK_INDICES)->getData(),
                                      memory.at(ARG_PAGED_SSM_BLOCK_INDICES_BEGINS)->getData(),
                                      memory.at(ARG_PAGED_SSM_NUM_PROCESSED_TOKENS)->getData(),
                                      memory.at(ARG_PAGED_SSM_CACHE_INTERVAL)->getData(),
                                      memory.at(ARG_PAGED_SSM_OUT)->getData(),
                                      shape,
                                      memory.at(ARG_PAGED_SSM_X)->getDescPtr()->getPrecision(),
                                      memory.at(ARG_PAGED_SSM_SUBSEQUENCE_BEGINS)->getDescPtr()->getPrecision(),
                                      m_state_scratch->getDataAs<float>(),
                                      m_scratch_head_dim,
                                      m_block_owners->getDataAs<int32_t>(),
                                      m_context->getCpuParallel());
}

impl_desc_type PagedSelectiveSSMExecutor::implType() const {
    return impl_desc_type::ref_any;
}

}  // namespace ov::intel_cpu
