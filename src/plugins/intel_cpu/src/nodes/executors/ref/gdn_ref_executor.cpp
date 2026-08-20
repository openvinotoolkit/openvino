// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/ref/gdn_ref_executor.hpp"

#include <cstddef>
#include <memory>
#include <utility>

#include "memory_desc/cpu_blocked_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/gated_delta_net_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/kernels/linear_attn/recurrent_linear_attn.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu {

bool GdnRefExecutor::supports(const GatedDeltaNetConfig& config) {
    const auto precision = config.descs.at(ARG_GDN_QUERY)->getPrecision();
    return precision == ov::element::f32 || precision == ov::element::f16 || precision == ov::element::bf16;
}

GdnRefExecutor::GdnRefExecutor(const GatedDeltaNetAttrs& attrs, const MemoryArgs& memory, ExecutorContext::CPtr context)
    : m_attrs(attrs),
      m_context(std::move(context)) {
    update(memory);
}

bool GdnRefExecutor::updateScratchpad(const MemoryArgs& memory) {
    const auto& queryShape = memory.at(ARG_GDN_QUERY)->getDescPtr()->getShape();
    if (queryShape.isDynamic()) {
        return true;
    }

    const auto& queryDims = queryShape.getStaticDims();
    OPENVINO_ASSERT(!queryDims.empty(), "GatedDeltaNet query tensor rank must be >= 1");
    const size_t headSize = queryDims.back();
    if (m_tmpInpBuffer != nullptr && m_cachedHeadSize == headSize) {
        return true;
    }

    const auto numWorkerThreads = m_context->getCpuParallel()->get_num_worker_threads();
    auto newMemDesc = std::make_shared<CpuBlockedMemoryDesc>(
        ov::element::f32,
        ov::intel_cpu::Shape{static_cast<size_t>(numWorkerThreads), 3 * headSize});
    m_tmpInpBuffer = m_context->getScratchPad()->createScratchPadMem(newMemDesc);
    m_cachedHeadSize = headSize;
    return m_tmpInpBuffer != nullptr;
}

bool GdnRefExecutor::update(const MemoryArgs& memory) {
    return updateScratchpad(memory);
}

void GdnRefExecutor::execute(const MemoryArgs& memory) {
    PlainTensor query(memory.at(ARG_GDN_QUERY));
    PlainTensor key(memory.at(ARG_GDN_KEY));
    PlainTensor value(memory.at(ARG_GDN_VALUE));
    PlainTensor recurrent_state(memory.at(ARG_GDN_STATE));
    PlainTensor gate(memory.at(ARG_GDN_GATE));
    PlainTensor beta(memory.at(ARG_GDN_BETA));
    PlainTensor output_attn(memory.at(ARG_GDN_OUT_ATTN));
    PlainTensor output_recurrent_state(memory.at(ARG_GDN_OUT_STATE));

    const size_t headSize = query.m_dims[3];
    if (m_tmpInpBuffer == nullptr || m_cachedHeadSize != headSize) {
        const auto numWorkerThreads = m_context->getCpuParallel()->get_num_worker_threads();
        auto newMemDesc = std::make_shared<CpuBlockedMemoryDesc>(
            ov::element::f32,
            ov::intel_cpu::Shape{static_cast<size_t>(numWorkerThreads), 3 * headSize});
        m_tmpInpBuffer = m_context->getScratchPad()->createScratchPadMem(newMemDesc);
        m_cachedHeadSize = headSize;
    }

    OPENVINO_ASSERT(m_tmpInpBuffer != nullptr, "GatedDeltaNet intrinsic executor scratchpad is not initialized");

    auto* temp_buffer = reinterpret_cast<float*>(m_tmpInpBuffer->getData());
    ov::Extensions::Cpu::XARCH::recurrent_linear_attn(query,
                                                      key,
                                                      value,
                                                      recurrent_state,
                                                      gate,
                                                      beta,
                                                      m_attrs.q_l2_norm_eps,
                                                      m_attrs.k_l2_norm_eps,
                                                      m_attrs.fuse_qk_l2norm,
                                                      output_attn,
                                                      output_recurrent_state,
                                                      temp_buffer,
                                                      m_context->getCpuParallel());
}

impl_desc_type GdnRefExecutor::implType() const {
    return impl_desc_type::ref_any;
}

}  // namespace ov::intel_cpu
