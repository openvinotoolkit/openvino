// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lora.h"

#include "nodes/input.h"
#include "cpu_memory.h"
#include "ov_ops/lora_subgraph.hpp"
#include "utils/debug_capabilities.h"
#include "shape_inference/shape_inference_pass_through.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

bool LoRA::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::op::internal::LoraSubgraph>(op)) {
            errorMessage = "Unknown LoRA operation : " + std::string(op->get_type_info().name) + " with name '" +
                           op->get_friendly_name() + "'";
        }
    } catch (...) {
        return false;
    }
    return true;
}

LoRA::LoRA(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, PassThroughShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto& loraModel = ov::as_type_ptr<ov::op::internal::LoraSubgraph>(op);
    OPENVINO_ASSERT(loraModel,
                    "Attempt to create LoRA node from an invalid op type: ",
                    op,
                    " with name ",
                    op->get_friendly_name());

    m_body = loraModel->get_function();
}

void LoRA::selectOptimalPrimitiveDescriptor() {
    // for the input configuration, just always use the parent configuration
    std::vector<PortConfig> inConfs;
    std::vector<Input::InputConfig> graphInputConfig;

    auto mainInputDesc = getParentOutputMemDesc(getParentEdgeAt(0));
    auto mainInputPrc = mainInputDesc->getPrecision(); // we have to align precision across all the inputs

    inConfs.emplace_back(mainInputDesc);
    // @todo should be always inplace after global memory reuse is fully supported by all the nodes
    bool isInPlace = context->memoryReuseGlobal();
    graphInputConfig.emplace_back(node::Input::InputConfig{mainInputDesc, isInPlace});

    for (size_t i = 1; i < getParentEdges().size(); i++) {
        auto desc = getParentOutputMemDesc(getParentEdgeAt(i))->cloneWithNewPrecision(mainInputPrc);
        inConfs.emplace_back(desc);
        graphInputConfig.emplace_back(node::Input::InputConfig{desc, isInPlace});
    }

    std::vector<Input::OutputConfig> graphOutputConfig;
    // enforce the same memory descriptor on the output as on the input to allow inPlace memory
    graphOutputConfig.emplace_back(node::Input::OutputConfig{inConfs.front().getMemDesc(), isInPlace});

    // configure the inner graph to get the information about output memory descriptors
    m_graph.Init(m_body, context, graphInputConfig, graphOutputConfig);

    // for the output descriptors, use the configuration of the graph's output nodes
    auto outputDescriptors = m_graph.getOutputMemoryDescriptors();

    const auto& desc = outputDescriptors.front();

    // just a sanity check
    CPU_NODE_ASSERT(desc->isCompatible(*(inConfs.front().getMemDesc())), "Unexpected input/output descriptor mismatch");

    std::vector<PortConfig> outConfs;

    outConfs.emplace_back(desc, BlockedMemoryDesc::FULL_MASK, 0); // use the memory from the first input inPlace

    const NodeConfig config(inConfs, outConfs);

    supportedPrimitiveDescriptors.clear();
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::undef);

    selectPrimitiveDescriptorByIndex(0);
}

int LoRA::registerToAllocationContext(int offset, AllocationContext& context) {
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        auto inputEdges = m_graph.GetInputNodesMap().at(i)->getChildEdgesAtPort(0);
        for (const auto& inputEdge : inputEdges) {
            OPENVINO_ASSERT(inputEdge->getStatus() == Edge::Status::Uninitialized,
                            "Expected Uninitialized Edge instead of: ", static_cast<int>(inputEdge->getStatus()));
            inputEdge->sharedMemFrom(parentEdge);
        }
    }

    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        auto childEdge = getChildEdgeAt(i);
        auto outputEdge = m_graph.GetOutputNodesMap().at(i)->getParentEdgeAt(0);
        outputEdge->sharedMemFrom(childEdge);
    }

    return m_graph.RegisterToAllocationContext(offset, context);
}

// @todo add ascii diagram for memory mapping / reuse
void LoRA::createPrimitive() {
    CPU_NODE_ASSERT(getOriginalInputsNumber() == m_graph.GetInputNodesMap().size(),
                    "Number of node inputs must be equal the number of inner graph's inputs");
    // Workaround to avoid making LoRa node always executable (isExecutable()) true
    // This way we update subgraph's input memory without performing an actual Infer() call
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        const auto& subgraphInputNode = m_graph.GetInputNodesMap().at(i);
        const auto& subgraphInputMemory = subgraphInputNode->getDstMemoryAtPort(0);
        auto mem = std::make_shared<Memory>(getEngine(), subgraphInputMemory->getDescPtr(), subgraphInputMemory->getMemoryBlock());
        subgraphMemoryPtrs.push_back(mem);
    }

    m_graph.Activate();
}

void LoRA::execute(dnnl::stream) {
    m_graph.Infer();
}

void LoRA::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void LoRA::prepareParams() {
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        // since the external and internal descriptors are compatible, we may pass the descriptor
        subgraphMemoryPtrs[i]->redefineDesc(getSrcMemoryAtPort(i)->getDescPtr());
    }
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
