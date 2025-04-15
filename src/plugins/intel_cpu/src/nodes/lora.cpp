// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lora.h"

#include "cpu_memory.h"
#include "nodes/input.h"
#include "ov_ops/lora_subgraph.hpp"
#include "shape_inference/shape_inference_pass_through.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu::node {

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
    CPU_NODE_ASSERT(loraModel,
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
    auto mainInputPrc = mainInputDesc->getPrecision();  // we have to align precision across all the inputs

    inConfs.emplace_back(mainInputDesc);

    constexpr bool isInPlace = true;
    graphInputConfig.emplace_back(node::Input::InputConfig{mainInputDesc, isInPlace});

    for (size_t i = 1; i < getParentEdges().size(); i++) {
        auto desc = getParentOutputMemDesc(getParentEdgeAt(i))->cloneWithNewPrecision(mainInputPrc);
        inConfs.emplace_back(desc);
        graphInputConfig.emplace_back(node::Input::InputConfig{desc, isInPlace});
    }

    std::vector<Input::OutputConfig> graphOutputConfig;
    // enforce the same memory descriptor on the output as on the input to allow inPlace memory
    graphOutputConfig.emplace_back(inConfs.front().getMemDesc(), isInPlace);

    // configure the inner graph to get the information about output memory descriptors
    m_graph.Init(m_body, context, graphInputConfig, graphOutputConfig);

    // for the output descriptors, use the configuration of the graph's output nodes
    auto outputDescriptors = m_graph.getOutputMemoryDescriptors();

    const auto& desc = outputDescriptors.front();

    // just a sanity check
    CPU_NODE_ASSERT(desc->isCompatible(*(inConfs.front().getMemDesc())), "Unexpected input/output descriptor mismatch");

    std::vector<PortConfig> outConfs;

    outConfs.emplace_back(desc, BlockedMemoryDesc::FULL_MASK, 0);  // use the memory from the first input inPlace

    const NodeConfig config(inConfs, outConfs);

    supportedPrimitiveDescriptors.clear();
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::undef);

    selectPrimitiveDescriptorByIndex(0);
}

int LoRA::registerToAllocationContext(int offset, AllocationContext& context) {
    CPU_NODE_ASSERT(getOriginalInputsNumber() == m_graph.inputsNumber(),
                    "Number of node inputs must be equal the number of inner graph's inputs");

    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        auto inputEdges = m_graph.getInputNodeByIndex(i)->getChildEdgesAtPort(0);
        for (const auto& inputEdge : inputEdges) {
            CPU_NODE_ASSERT(inputEdge->getStatus() == Edge::Status::Uninitialized,
                            "Expected Uninitialized Edge instead of: ",
                            static_cast<int>(inputEdge->getStatus()));
            inputEdge->sharedMemFrom(parentEdge);
        }
    }

    CPU_NODE_ASSERT(getOriginalOutputsNumber() == m_graph.outputsNumber(),
                    "Number of node outputs must be equal the number of inner graph's outputs");

    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        auto childEdge = getChildEdgeAt(i);
        auto outputEdge = m_graph.getOutputNodeByIndex(i)->getParentEdgeAt(0);
        outputEdge->sharedMemFrom(childEdge);
    }

    return m_graph.RegisterToAllocationContext(offset, context);
}

void LoRA::createPrimitive() {
    CPU_NODE_ASSERT(getOriginalInputsNumber() == m_graph.inputsNumber(),
                    "Number of node inputs must be equal the number of inner graph's inputs");
    // Workaround to avoid making LoRa node always executable (isExecutable() = true)
    // This way we update subgraph's input memory without performing an actual Infer() call
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        auto subgraphInputNode = m_graph.getInputNodeByIndex(i);
        auto subgraphInputMemory = subgraphInputNode->getDstMemoryAtPort(0);
        subgraphMemoryPtrs.emplace_back(subgraphInputMemory);
    }

    m_graph.Activate();
}

void LoRA::execute(const dnnl::stream&) {
    m_graph.Infer();
}

void LoRA::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void LoRA::prepareParams() {
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        // since the external and internal descriptors are compatible, we may pass the descriptor
        subgraphMemoryPtrs[i]->redefineDesc(getSrcMemoryAtPort(i)->getDescPtr());
    }
}

}  // namespace ov::intel_cpu::node
