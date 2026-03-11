// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_delta_net.h"

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "graph_context.h"
#include "kernels/linear_attn/recurrent_linear_attn.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/plain_tensor.hpp"

using namespace ov::Extensions::Cpu;
using namespace ov::Extensions::Cpu::XARCH;

namespace ov::intel_cpu::node {

GatedDeltaNet::GatedDeltaNet(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto& gdn = ov::as_type_ptr<ov::op::GatedDeltaNet>(op);
    m_fuse_q_scale = gdn->get_config().fuse_q_scale;
    m_fuse_qk_l2norm = gdn->get_config().fuse_qk_l2norm;
    m_q_l2_norm_eps = gdn->get_config().q_l2_norm_eps;
    m_k_l2_norm_eps = gdn->get_config().k_l2_norm_eps;
}

void GatedDeltaNet::initSupportedPrimitiveDescriptors() {
    // TODO: support other precision CVS-182464
    auto dataPrecision = ov::element::f32;
    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        inPortConfigs.emplace_back(LayoutType::ncsp, dataPrecision, getInputShapeAtPort(i), false, -1);
    }
    std::vector<PortConfigurator> outPortConfigs = {
        PortConfigurator{LayoutType::ncsp, dataPrecision, getOutputShapeAtPort(0), false, -1},
        PortConfigurator{LayoutType::ncsp, dataPrecision, getOutputShapeAtPort(1), false, -1}};
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void GatedDeltaNet::createPrimitive() {
    const auto queryDims = getInputShapeAtPort(0).getDims();
    auto headSize = *(queryDims.end() - 1);
    const auto numWorkerThreads = context->getCpuParallel()->get_num_worker_threads();
    auto newMemDesc = std::make_shared<CpuBlockedMemoryDesc>(
        ov::element::f32,
        ov::intel_cpu::Shape{static_cast<size_t>(numWorkerThreads), 3 * headSize});
    m_tmpInpBuffer = context->getScratchPad()->createScratchPadMem(newMemDesc);
}

void GatedDeltaNet::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto originalInputNumber = getOriginalInputsNumber();
    std::vector<MemoryPtr> inputs(originalInputNumber);
    std::vector<MemoryPtr> outputs(2);

    for (size_t i = 0; i < originalInputNumber; i++) {
        inputs[i] = getSrcMemoryAtPort(i);
    }

    outputs[0] = getDstMemoryAtPort(0);
    outputs[1] = getDstMemoryAtPort(1);

    PlainTensor query(inputs[0]);
    PlainTensor key(inputs[1]);
    PlainTensor value(inputs[2]);
    PlainTensor recurrent_state(inputs[3]);
    PlainTensor gate(inputs[4]);
    PlainTensor beta(inputs[5]);
    PlainTensor output_attn(outputs[0]);
    PlainTensor output_recurrent_state(outputs[1]);

    auto* temp_buffer = m_tmpInpBuffer->getDataAs<float>();
    recurrent_linear_attn(query,
                          key,
                          value,
                          recurrent_state,
                          gate,
                          beta,
                          m_q_l2_norm_eps,
                          m_k_l2_norm_eps,
                          m_fuse_qk_l2norm,
                          m_fuse_q_scale,
                          output_attn,
                          output_recurrent_state,
                          temp_buffer,
                          context->getCpuParallel());
}

bool GatedDeltaNet::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                         std::string& errorMessage) noexcept {
    if (op == nullptr || !ov::is_type<ov::op::GatedDeltaNet>(op)) {
        errorMessage = "Node is not an instance of ov::op::GatedDeltaNet.";
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
