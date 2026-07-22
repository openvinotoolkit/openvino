// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_delta_net.h"

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/gated_delta_net_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {

GatedDeltaNet::GatedDeltaNet(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto& gdn = ov::as_type_ptr<ov::op::internal::GatedDeltaNet>(op);
    m_fuse_qk_l2norm = gdn->get_fuse_qk_l2norm();
    m_q_l2_norm_eps = gdn->get_q_l2_norm_eps();
    m_k_l2_norm_eps = gdn->get_k_l2_norm_eps();
    m_attrs.fuse_qk_l2norm = m_fuse_qk_l2norm;
    m_attrs.q_l2_norm_eps = m_q_l2_norm_eps;
    m_attrs.k_l2_norm_eps = m_k_l2_norm_eps;
    m_atoi[ARG_GDN_QUERY] = 0;
    m_atoi[ARG_GDN_KEY] = 1;
    m_atoi[ARG_GDN_VALUE] = 2;
    m_atoi[ARG_GDN_STATE] = 3;
    m_atoi[ARG_GDN_GATE] = 4;
    m_atoi[ARG_GDN_BETA] = 5;
}

void GatedDeltaNet::initSupportedPrimitiveDescriptors() {
    auto dataPrecision = getOriginalOutputPrecisionAtPort(0);
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();

    MemoryDescArgs descs;
    for (const auto& [argId, portId] : m_atoi) {
        descs[argId] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(dataPrecision, getInputShapeAtPort(portId));
    }
    descs[ARG_GDN_OUT_ATTN] =
        creatorsMap.at(LayoutType::ncsp)->createSharedDesc(dataPrecision, getOutputShapeAtPort(0));
    descs[ARG_GDN_OUT_STATE] =
        creatorsMap.at(LayoutType::ncsp)->createSharedDesc(dataPrecision, getOutputShapeAtPort(1));

    auto executionContext = std::make_shared<ExecutorContext>(context, getImplPriority(), privateWeightCache);
    m_factory = std::make_shared<ExecutorFactory<GatedDeltaNetAttrs>>(m_attrs, executionContext, descs);

    const auto nodeDescriptorsList = m_factory->getProperMemoryDescriptors(descs);
    for (const auto& nodeDescriptors : nodeDescriptorsList) {
        NodeConfig nodeConfig;
        nodeConfig.inConfs.resize(getParentEdges().size());

        for (const auto& [argId, portId] : m_atoi) {
            if (nodeDescriptors.count(argId)) {
                nodeConfig.inConfs[portId] = PortConfig{nodeDescriptors.at(argId)};
            }
        }

        nodeConfig.outConfs.emplace_back(nodeDescriptors.at(ARG_GDN_OUT_ATTN));
        nodeConfig.outConfs.emplace_back(nodeDescriptors.at(ARG_GDN_OUT_STATE));

        supportedPrimitiveDescriptors.emplace_back(nodeConfig, impl_desc_type::undef);
    }
}

void GatedDeltaNet::createPrimitive() {
    for (const auto& [argId, portId] : m_atoi) {
        m_memory[argId] = getSrcMemoryAtPort(portId);
    }
    m_memory[ARG_GDN_OUT_ATTN] = getDstMemoryAtPort(0);
    m_memory[ARG_GDN_OUT_STATE] = getDstMemoryAtPort(1);

    m_executor = m_factory->make(m_memory);

    Node::createPrimitive();
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());
}

void GatedDeltaNet::prepareParams() {
    for (const auto& [argId, portId] : m_atoi) {
        m_memory[argId] = getSrcMemoryAtPort(portId);
    }
    m_memory[ARG_GDN_OUT_ATTN] = getDstMemoryAtPort(0);
    m_memory[ARG_GDN_OUT_STATE] = getDstMemoryAtPort(1);

    m_executor->update(m_memory);
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->implType());
}

void GatedDeltaNet::execute([[maybe_unused]] const dnnl::stream& strm) {
    for (const auto& [argId, portId] : m_atoi) {
        m_memory[argId] = getSrcMemoryAtPort(portId);
    }
    m_memory[ARG_GDN_OUT_ATTN] = getDstMemoryAtPort(0);
    m_memory[ARG_GDN_OUT_STATE] = getDstMemoryAtPort(1);

    m_executor->execute(m_memory);
}

bool GatedDeltaNet::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                         std::string& errorMessage) noexcept {
    if (op == nullptr || !ov::is_type<ov::op::internal::GatedDeltaNet>(op)) {
        errorMessage = "Node is not an instance of ov::op::internal::GatedDeltaNet.";
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
