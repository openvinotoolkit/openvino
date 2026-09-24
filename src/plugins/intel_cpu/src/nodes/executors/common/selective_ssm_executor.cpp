// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm_executor.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <utility>

#include "memory_desc/cpu_blocked_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/selective_ssm_config.hpp"
#include "nodes/kernels/selective_ssm.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {
namespace {

bool is_supported_precision(const ov::element::Type& precision) {
    return any_of(precision, ov::element::f32, ov::element::f16, ov::element::bf16);
}

}  // namespace

bool SelectiveSSMExecutor::supports(const SelectiveSSMConfig& config) {
    const auto precision = config.descs.at(ARG_SSM_A)->getPrecision();
    if (!is_supported_precision(precision)) {
        return false;
    }
    constexpr std::array args{
        ARG_SSM_DT,
        ARG_SSM_B,
        ARG_SSM_X,
        ARG_SSM_C,
        ARG_SSM_STATE,
        ARG_SSM_OUT,
        ARG_SSM_OUT_STATE,
    };
    return std::all_of(args.begin(), args.end(), [&](const auto arg) {
        return config.descs.at(arg)->getPrecision() == precision;
    });
}

SelectiveSSMExecutor::SelectiveSSMExecutor(const SelectiveSSMAttrs& /*attrs*/,
                                           const MemoryArgs& memory,
                                           ExecutorContext::CPtr context)
    : m_context(std::move(context)) {
    update(memory);
}

bool SelectiveSSMExecutor::update_scratchpad(const MemoryArgs& memory) {
    const auto& x_shape = memory.at(ARG_SSM_X)->getDescPtr()->getShape();
    const auto& B_shape = memory.at(ARG_SSM_B)->getDescPtr()->getShape();
    const auto& state_shape = memory.at(ARG_SSM_STATE)->getDescPtr()->getShape();
    if (x_shape.isDynamic() || B_shape.isDynamic() || state_shape.isDynamic()) {
        return true;
    }
    const auto& x_dims = x_shape.getStaticDims();
    const auto& B_dims = B_shape.getStaticDims();
    const auto& state_dims = state_shape.getStaticDims();
    const auto precision = memory.at(ARG_SSM_X)->getDescPtr()->getPrecision();
    OPENVINO_ASSERT(x_dims.size() == 4 && B_dims.size() == 4 && state_dims.size() == 4);
    const auto head_dim = x_dims[3];
    const auto state_size = state_dims[3];
    const auto thread_count = static_cast<size_t>(m_context->getCpuParallel()->get_num_worker_threads());
    const auto outer_work = node::kernel::checked_size_product({x_dims[0], x_dims[2]}, "outer work items");
    const auto scratch_head_dim = node::kernel::get_scratch_head_dim(head_dim, state_size, outer_work, thread_count);
    const auto needs_state_scratch = precision != ov::element::f32;
    const auto state_scratch_elements =
        needs_state_scratch
            ? node::kernel::checked_size_product({thread_count, scratch_head_dim, state_size}, "state scratch")
            : size_t{0};
    const auto projection_elements =
        node::kernel::checked_size_product({B_dims[0], B_dims[1], B_dims[2], B_dims[3]}, "B/C projection");
    const auto projection_scratch_elements =
        precision == ov::element::f32
            ? size_t{0}
            : node::kernel::checked_size_product({size_t{2}, projection_elements}, "B/C projection scratch");
    if (m_scratch && m_scratch_head_dim == scratch_head_dim && m_scratch_state_size == state_size &&
        m_state_scratch_elements == state_scratch_elements &&
        m_projection_scratch_elements == projection_scratch_elements &&
        m_cached_projection_elements == projection_elements) {
        return true;
    }

    const auto total_scratch_elements =
        node::kernel::checked_size_sum({state_scratch_elements, projection_scratch_elements},
                                       "combined state and B/C projection scratch");
    const auto desc =
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::f32,
                                               ov::intel_cpu::Shape{std::max(size_t{1}, total_scratch_elements)});
    m_scratch = m_context->getScratchPad()->createScratchPadMem(desc);
    m_scratch_head_dim = scratch_head_dim;
    m_scratch_state_size = state_size;
    m_state_scratch_elements = state_scratch_elements;
    m_projection_scratch_elements = projection_scratch_elements;
    m_cached_projection_elements = projection_elements;
    return m_scratch != nullptr;
}

bool SelectiveSSMExecutor::update(const MemoryArgs& memory) {
    return update_scratchpad(memory);
}

void SelectiveSSMExecutor::execute(const MemoryArgs& memory) {
    const auto& x_dims = memory.at(ARG_SSM_X)->getStaticDims();
    const auto& B_dims = memory.at(ARG_SSM_B)->getStaticDims();
    const auto& state_dims = memory.at(ARG_SSM_STATE)->getStaticDims();
    OPENVINO_ASSERT(x_dims.size() == 4 && B_dims.size() == 4 && state_dims.size() == 4);
    const auto precision = memory.at(ARG_SSM_X)->getDescPtr()->getPrecision();
    const auto projection_elements =
        node::kernel::checked_size_product({B_dims[0], B_dims[1], B_dims[2], B_dims[3]}, "B/C projection");
    const auto thread_count = static_cast<size_t>(m_context->getCpuParallel()->get_num_worker_threads());
    const auto outer_work = node::kernel::checked_size_product({x_dims[0], x_dims[2]}, "outer work items");
    const auto expected_scratch_head_dim =
        node::kernel::get_scratch_head_dim(x_dims[3], state_dims[3], outer_work, thread_count);
    const auto expected_state_scratch_elements =
        precision != ov::element::f32
            ? node::kernel::checked_size_product({thread_count, expected_scratch_head_dim, state_dims[3]},
                                                 "state scratch")
            : size_t{0};
    const auto expected_projection_scratch_elements =
        precision == ov::element::f32
            ? size_t{0}
            : node::kernel::checked_size_product({size_t{2}, projection_elements}, "B/C projection scratch");
    if (!m_scratch || m_scratch_state_size != state_dims[3] || m_scratch_head_dim != expected_scratch_head_dim ||
        m_state_scratch_elements != expected_state_scratch_elements ||
        m_projection_scratch_elements != expected_projection_scratch_elements ||
        m_cached_projection_elements != projection_elements) {
        OPENVINO_ASSERT(update_scratchpad(memory));
    }
    OPENVINO_ASSERT(m_scratch != nullptr, "SelectiveSSM scratch memory is not initialized.");

    node::kernel::SelectiveSSMShape shape{x_dims[0], x_dims[1], x_dims[2], x_dims[3], B_dims[2], B_dims[3]};
    auto* state_scratch = m_scratch->getDataAs<float>();
    const float* converted_B = nullptr;
    const float* converted_C = nullptr;
    if (precision != ov::element::f32) {
        auto* projection_scratch = state_scratch + m_state_scratch_elements;
        converted_B = projection_scratch;
        converted_C = projection_scratch + projection_elements;
        if (projection_elements > 0) {
            cpu_parallel_convert(memory.at(ARG_SSM_B)->getData(),
                                 projection_scratch,
                                 precision,
                                 ov::element::f32,
                                 projection_elements);
            cpu_parallel_convert(memory.at(ARG_SSM_C)->getData(),
                                 projection_scratch + projection_elements,
                                 precision,
                                 ov::element::f32,
                                 projection_elements);
        }
    }
    node::kernel::selective_ssm(memory.at(ARG_SSM_A)->getData(),
                                memory.at(ARG_SSM_DT)->getData(),
                                memory.at(ARG_SSM_B)->getData(),
                                memory.at(ARG_SSM_X)->getData(),
                                memory.at(ARG_SSM_C)->getData(),
                                memory.at(ARG_SSM_STATE)->getData(),
                                memory.at(ARG_SSM_OUT)->getData(),
                                memory.at(ARG_SSM_OUT_STATE)->getData(),
                                shape,
                                precision,
                                state_scratch,
                                m_scratch_head_dim,
                                m_context->getCpuParallel(),
                                converted_B,
                                converted_C);
}

impl_desc_type SelectiveSSMExecutor::implType() const {
    return impl_desc_type::ref_any;
}

}  // namespace ov::intel_cpu
