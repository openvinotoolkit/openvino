// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_gated_delta_net.h"

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "graph_context.h"
#include "kernels/linear_attn/recurrent_linear_attn.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_gated_delta_net.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/plain_tensor.hpp"

using namespace ov::Extensions::Cpu;
using namespace ov::Extensions::Cpu::XARCH;

namespace ov::intel_cpu::node {

PagedGatedDeltaNet::PagedGatedDeltaNet(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto& pgdn = ov::as_type_ptr<ov::op::internal::PagedGatedDeltaNet>(op);
    OPENVINO_ASSERT(pgdn != nullptr, "Expected PagedGatedDeltaNet node");
    m_use_qk_l2norm = pgdn->get_use_qk_l2norm();
    m_q_l2_norm_eps = pgdn->get_q_l2_norm_eps();
    m_k_l2_norm_eps = pgdn->get_k_l2_norm_eps();
}

void PagedGatedDeltaNet::initSupportedPrimitiveDescriptors() {
    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        inPortConfigs.emplace_back(LayoutType::ncsp,
                                   getOriginalInputPrecisionAtPort(i),
                                   getInputShapeAtPort(i),
                                   false,
                                   -1);
    }

    std::vector<PortConfigurator> outPortConfigs = {
        PortConfigurator{LayoutType::ncsp, getOriginalOutputPrecisionAtPort(0), getOutputShapeAtPort(0), false, -1}};
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void PagedGatedDeltaNet::createPrimitive() {
    const auto queryDims = getInputShapeAtPort(0).getDims();
    auto headSize = *(queryDims.end() - 1);
    const auto numWorkerThreads = context->getCpuParallel()->get_num_worker_threads();
    auto newMemDesc = std::make_shared<CpuBlockedMemoryDesc>(
        ov::element::f32,
        ov::intel_cpu::Shape{static_cast<size_t>(numWorkerThreads), 3 * headSize});
    m_tmpInpBuffer = context->getScratchPad()->createScratchPadMem(newMemDesc);
}

void PagedGatedDeltaNet::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto originalInputNumber = getOriginalInputsNumber();
    std::vector<MemoryPtr> inputs(originalInputNumber);

    for (size_t i = 0; i < originalInputNumber; i++) {
        inputs[i] = getSrcMemoryAtPort(i);
    }

    auto output = getDstMemoryAtPort(0);

    PlainTensor query(inputs[0]);
    PlainTensor key(inputs[1]);
    PlainTensor value(inputs[2]);
    PlainTensor recurrent_state(inputs[3]);
    PlainTensor gate(inputs[4]);
    PlainTensor beta(inputs[5]);

    PlainTensor subsequence_begins(inputs[6]);
    PlainTensor block_indices(inputs[7]);
    PlainTensor block_indices_begins(inputs[8]);
    PlainTensor past_lens(inputs[9]);
    PlainTensor cache_interval(inputs[10]);
    PlainTensor output_attn(output);

    auto* temp_buffer = m_tmpInpBuffer->getDataAs<float>();

    recurrent_linear_attn_paged(query,
                                key,
                                value,
                                recurrent_state,
                                gate,
                                beta,
                                subsequence_begins,
                                block_indices,
                                block_indices_begins,
                                past_lens,
                                cache_interval,
                                m_q_l2_norm_eps,
                                m_k_l2_norm_eps,
                                m_use_qk_l2norm,
                                output_attn,
                                temp_buffer,
                                context->getCpuParallel());
}

bool PagedGatedDeltaNet::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                              std::string& errorMessage) noexcept {
    if (op == nullptr || !ov::is_type<ov::op::internal::PagedGatedDeltaNet>(op)) {
        errorMessage = "Node is not an instance of ov::op::internal::PagedGatedDeltaNet.";
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
