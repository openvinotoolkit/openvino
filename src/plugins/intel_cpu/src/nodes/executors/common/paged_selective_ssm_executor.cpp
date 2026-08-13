// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_selective_ssm_executor.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>

#include "memory_desc/cpu_blocked_memory_desc.h"
#include "nodes/common/cpu_convert.h"
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
    const auto& B_shape = memory.at(ARG_PAGED_SSM_B)->getDescPtr()->getShape();
    const auto& state_shape = memory.at(ARG_PAGED_SSM_STATE)->getDescPtr()->getShape();
    const auto& subsequence_shape = memory.at(ARG_PAGED_SSM_SUBSEQUENCE_BEGINS)->getDescPtr()->getShape();
    if (x_shape.isDynamic() || B_shape.isDynamic() || state_shape.isDynamic() || subsequence_shape.isDynamic()) {
        return true;
    }
    const auto& x_dims = x_shape.getStaticDims();
    const auto& B_dims = B_shape.getStaticDims();
    const auto& state_dims = state_shape.getStaticDims();
    const auto& subsequence_dims = subsequence_shape.getStaticDims();
    const auto precision = memory.at(ARG_PAGED_SSM_X)->getDescPtr()->getPrecision();
    OPENVINO_ASSERT(x_dims.size() == 3 && B_dims.size() == 3 && state_dims.size() == 4 &&
                    subsequence_dims.size() == 1 && subsequence_dims[0] >= 1);
    const auto head_dim = x_dims[2];
    const auto state_size = state_dims[3];
    const auto physical_blocks = state_dims[0];
    const node::kernel::PagedSelectiveSSMShape
        shape{x_dims[0], x_dims[1], head_dim, B_dims[1], state_size, physical_blocks, 0, 0};
    node::kernel::validate_paged_selective_ssm_shape(shape);
    const auto thread_count = static_cast<size_t>(m_context->getCpuParallel()->get_num_worker_threads());
    const auto sequence_count = subsequence_dims[0] - 1;
    const auto outer_work = node::kernel::checked_size_product({sequence_count, x_dims[1]}, "outer work items");
    const auto scratch_head_dim = node::kernel::get_scratch_head_dim(head_dim, state_size, outer_work, thread_count);
    const auto scratch_elements =
        node::kernel::checked_size_product({scratch_head_dim, state_size}, "state scratch per worker");
    const auto state_scratch_elements =
        node::kernel::checked_size_product({thread_count, scratch_elements}, "state scratch");
    const auto projection_elements =
        node::kernel::checked_size_product({B_dims[0], B_dims[1], B_dims[2]}, "B/C projection");
    const auto projection_scratch_elements =
        precision == ov::element::f32
            ? size_t{0}
            : node::kernel::checked_size_product({size_t{2}, projection_elements}, "B/C projection scratch");
    if (m_scratch && m_scratch_head_dim == scratch_head_dim && m_scratch_state_size == state_size &&
        m_state_scratch_elements == state_scratch_elements &&
        m_projection_scratch_elements == projection_scratch_elements && m_cached_physical_blocks == physical_blocks &&
        m_cached_projection_elements == projection_elements) {
        return true;
    }

    static_assert(sizeof(float) == sizeof(int32_t) && alignof(float) == alignof(int32_t));
    const auto total_scratch_elements =
        node::kernel::checked_size_sum({state_scratch_elements, projection_scratch_elements, physical_blocks},
                                       "combined state, B/C projection, and block-owner scratch");
    const auto scratch_desc =
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::f32,
                                               ov::intel_cpu::Shape{std::max(size_t{1}, total_scratch_elements)});
    m_scratch = m_context->getScratchPad()->createScratchPadMem(scratch_desc);
    m_scratch_head_dim = scratch_head_dim;
    m_scratch_state_size = state_size;
    m_state_scratch_elements = state_scratch_elements;
    m_projection_scratch_elements = projection_scratch_elements;
    m_cached_physical_blocks = physical_blocks;
    m_cached_projection_elements = projection_elements;
    return m_scratch != nullptr;
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
    const auto precision = memory.at(ARG_PAGED_SSM_X)->getDescPtr()->getPrecision();
    const auto projection_elements =
        node::kernel::checked_size_product({B_dims[0], B_dims[1], B_dims[2]}, "B/C projection");
    const auto sequence_count = subsequence_dims[0] - 1;
    const auto thread_count = static_cast<size_t>(m_context->getCpuParallel()->get_num_worker_threads());
    const auto outer_work = node::kernel::checked_size_product({sequence_count, x_dims[1]}, "outer work items");
    const auto expected_scratch_head_dim =
        node::kernel::get_scratch_head_dim(x_dims[2], state_dims[3], outer_work, thread_count);
    const auto expected_projection_scratch_elements =
        precision == ov::element::f32
            ? size_t{0}
            : node::kernel::checked_size_product({size_t{2}, projection_elements}, "B/C projection scratch");
    OPENVINO_ASSERT(block_begins_dims.size() == 1 && processed_dims.size() == 1 && interval_dims.size() == 1 &&
                        block_begins_dims[0] == sequence_count + 1 && processed_dims[0] == sequence_count &&
                        interval_dims[0] == sequence_count,
                    "PagedSelectiveSSM metadata tensor lengths are inconsistent.");
    if (!m_scratch || m_scratch_state_size != state_dims[3] || m_scratch_head_dim != expected_scratch_head_dim ||
        m_cached_physical_blocks != state_dims[0] ||
        m_projection_scratch_elements != expected_projection_scratch_elements ||
        m_cached_projection_elements != projection_elements) {
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
    auto* state_scratch = m_scratch->getDataAs<float>();
    const float* converted_B = nullptr;
    const float* converted_C = nullptr;
    if (precision != ov::element::f32 && projection_elements > 0) {
        auto* projection_scratch = state_scratch + m_state_scratch_elements;
        converted_B = projection_scratch;
        converted_C = projection_scratch + projection_elements;
        cpu_parallel_convert(memory.at(ARG_PAGED_SSM_B)->getData(),
                             projection_scratch,
                             precision,
                             ov::element::f32,
                             projection_elements);
        cpu_parallel_convert(memory.at(ARG_PAGED_SSM_C)->getData(),
                             projection_scratch + projection_elements,
                             precision,
                             ov::element::f32,
                             projection_elements);
    }
    auto* block_owners =
        reinterpret_cast<int32_t*>(state_scratch + m_state_scratch_elements + m_projection_scratch_elements);
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
                                      precision,
                                      memory.at(ARG_PAGED_SSM_SUBSEQUENCE_BEGINS)->getDescPtr()->getPrecision(),
                                      state_scratch,
                                      m_scratch_head_dim,
                                      block_owners,
                                      m_context->getCpuParallel(),
                                      converted_B,
                                      converted_C);
}

impl_desc_type PagedSelectiveSSMExecutor::implType() const {
    return impl_desc_type::ref_any;
}

}  // namespace ov::intel_cpu
