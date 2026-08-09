// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "selective_ssm.h"

#include <memory>
#include <vector>

#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/selective_ssm_config.hpp"
#include "nodes/node_config.h"
#include "openvino/core/except.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {

SelectiveSSM::SelectiveSSM(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    m_atoi[ARG_SSM_A] = 0;
    m_atoi[ARG_SSM_DT] = 1;
    m_atoi[ARG_SSM_B] = 2;
    m_atoi[ARG_SSM_X] = 3;
    m_atoi[ARG_SSM_C] = 4;
    m_atoi[ARG_SSM_STATE] = 5;
}

void SelectiveSSM::initSupportedPrimitiveDescriptors() {
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();

    MemoryDescArgs descs;
    for (const auto& [argId, portId] : m_atoi) {
        descs[argId] = creatorsMap.at(LayoutType::ncsp)
                           ->createSharedDesc(getOriginalInputPrecisionAtPort(portId), getInputShapeAtPort(portId));
    }
    descs[ARG_SSM_OUT] = creatorsMap.at(LayoutType::ncsp)
                             ->createSharedDesc(getOriginalOutputPrecisionAtPort(0), getOutputShapeAtPort(0));
    descs[ARG_SSM_OUT_STATE] = creatorsMap.at(LayoutType::ncsp)
                                   ->createSharedDesc(getOriginalOutputPrecisionAtPort(1), getOutputShapeAtPort(1));

    auto executionContext = std::make_shared<ExecutorContext>(context, getImplPriority(), privateWeightCache);
    m_factory = std::make_shared<ExecutorFactory<SelectiveSSMAttrs>>(m_attrs, executionContext, descs);

    const auto nodeDescriptorsList = m_factory->getProperMemoryDescriptors(descs);
    for (const auto& nodeDescriptors : nodeDescriptorsList) {
        NodeConfig nodeConfig;
        nodeConfig.inConfs.resize(getParentEdges().size());

        for (const auto& [argId, portId] : m_atoi) {
            if (nodeDescriptors.count(argId)) {
                nodeConfig.inConfs[portId] = PortConfig{nodeDescriptors.at(argId)};
            }
        }

        nodeConfig.outConfs.emplace_back(nodeDescriptors.at(ARG_SSM_OUT));
        nodeConfig.outConfs.emplace_back(nodeDescriptors.at(ARG_SSM_OUT_STATE));
        supportedPrimitiveDescriptors.emplace_back(nodeConfig, impl_desc_type::undef);
    }
}

void SelectiveSSM::createPrimitive() {
    for (const auto& [argId, portId] : m_atoi) {
        m_memory[argId] = getSrcMemoryAtPort(portId);
    }
    m_memory[ARG_SSM_OUT] = getDstMemoryAtPort(0);
    m_memory[ARG_SSM_OUT_STATE] = getDstMemoryAtPort(1);

    m_executor = m_factory->make(m_memory);
    Node::createPrimitive();
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());
}

void SelectiveSSM::prepareParams() {
    for (const auto& [argId, portId] : m_atoi) {
        m_memory[argId] = getSrcMemoryAtPort(portId);
    }
    m_memory[ARG_SSM_OUT] = getDstMemoryAtPort(0);
    m_memory[ARG_SSM_OUT_STATE] = getDstMemoryAtPort(1);

    m_executor->update(m_memory);
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());
}

void SelectiveSSM::execute([[maybe_unused]] const dnnl::stream& strm) {
    for (const auto& [argId, portId] : m_atoi) {
        m_memory[argId] = getSrcMemoryAtPort(portId);
    }
    m_memory[ARG_SSM_OUT] = getDstMemoryAtPort(0);
    m_memory[ARG_SSM_OUT_STATE] = getDstMemoryAtPort(1);

    m_executor->execute(m_memory);
}

bool SelectiveSSM::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    if (op == nullptr || !ov::is_type<ov::op::internal::SelectiveSSM>(op)) {
        errorMessage = "Node is not an instance of ov::op::internal::SelectiveSSM.";
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
