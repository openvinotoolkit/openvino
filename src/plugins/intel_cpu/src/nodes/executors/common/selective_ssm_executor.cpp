// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm_executor.hpp"

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

constexpr size_t target_scratch_elements = 8192;

bool is_supported_precision(const ov::element::Type& precision) {
    return precision == ov::element::f32 || precision == ov::element::f16 || precision == ov::element::bf16;
}

}  // namespace

bool SelectiveSSMExecutor::supports(const SelectiveSSMConfig& config) {
    const auto precision = config.descs.at(ARG_SSM_A)->getPrecision();
    if (!is_supported_precision(precision)) {
        return false;
    }
    for (const auto arg :
         {ARG_SSM_DT, ARG_SSM_B, ARG_SSM_X, ARG_SSM_C, ARG_SSM_STATE, ARG_SSM_OUT, ARG_SSM_OUT_STATE}) {
        if (config.descs.at(arg)->getPrecision() != precision) {
            return false;
        }
    }
    return true;
}

SelectiveSSMExecutor::SelectiveSSMExecutor(const SelectiveSSMAttrs&,
                                           const MemoryArgs& memory,
                                           ExecutorContext::CPtr context)
    : m_context(std::move(context)) {
    update(memory);
}

bool SelectiveSSMExecutor::update_scratchpad(const MemoryArgs& memory) {
    const auto& x_shape = memory.at(ARG_SSM_X)->getDescPtr()->getShape();
    const auto& state_shape = memory.at(ARG_SSM_STATE)->getDescPtr()->getShape();
    if (x_shape.isDynamic() || state_shape.isDynamic()) {
        return true;
    }
    const auto& x_dims = x_shape.getStaticDims();
    const auto& state_dims = state_shape.getStaticDims();
    OPENVINO_ASSERT(x_dims.size() == 4 && state_dims.size() == 4);
    const auto head_dim = x_dims[3];
    const auto state_size = state_dims[3];
    OPENVINO_ASSERT(state_size > 0, "SelectiveSSM state_size must be greater than zero.");
    const auto scratch_head_dim = std::max(size_t{1}, std::min(head_dim, target_scratch_elements / state_size));
    if (m_state_scratch && m_scratch_head_dim == scratch_head_dim && m_scratch_state_size == state_size) {
        return true;
    }

    const auto thread_count = static_cast<size_t>(m_context->getCpuParallel()->get_num_worker_threads());
    const auto desc =
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::f32,
                                               ov::intel_cpu::Shape{thread_count, scratch_head_dim * state_size});
    m_state_scratch = m_context->getScratchPad()->createScratchPadMem(desc);
    m_scratch_head_dim = scratch_head_dim;
    m_scratch_state_size = state_size;
    return m_state_scratch != nullptr;
}

bool SelectiveSSMExecutor::update(const MemoryArgs& memory) {
    return update_scratchpad(memory);
}

void SelectiveSSMExecutor::execute(const MemoryArgs& memory) {
    const auto& x_dims = memory.at(ARG_SSM_X)->getStaticDims();
    const auto& B_dims = memory.at(ARG_SSM_B)->getStaticDims();
    const auto& state_dims = memory.at(ARG_SSM_STATE)->getStaticDims();
    OPENVINO_ASSERT(x_dims.size() == 4 && B_dims.size() == 4 && state_dims.size() == 4);
    if (!m_state_scratch || m_scratch_state_size != state_dims[3] || m_scratch_head_dim > x_dims[3]) {
        OPENVINO_ASSERT(update_scratchpad(memory));
    }

    node::kernel::SelectiveSSMShape shape{x_dims[0], x_dims[1], x_dims[2], x_dims[3], B_dims[2], B_dims[3]};
    const auto precision = memory.at(ARG_SSM_X)->getDescPtr()->getPrecision();
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
                                m_state_scratch->getDataAs<float>(),
                                m_scratch_head_dim,
                                m_context->getCpuParallel());
}

impl_desc_type SelectiveSSMExecutor::implType() const {
    return impl_desc_type::ref_any;
}

}  // namespace ov::intel_cpu
