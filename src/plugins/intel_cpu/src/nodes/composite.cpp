// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "composite.h"

#include "cpu_memory.h"
#include "nodes/input.h"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "transformations/cpu_opset/common/op/submodel.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu::node {

bool Composite::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::intel_cpu::SubModel>(op)) {
            errorMessage = "Unknown SubGraph operation : " + std::string(op->get_type_info().name) + " with name '" +
                           op->get_friendly_name() + "'";
        }
    } catch (...) {
        return false;
    }
    return true;
}

Composite::Composite(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto& subModel = ov::as_type_ptr<SubModel>(op);
    CPU_NODE_ASSERT(subModel, "Attempt to create SubGraph node from an invalid op type: ", op);

    m_body = subModel->get_function();
}

void Composite::selectOptimalPrimitiveDescriptor() {
    // for the input configuration, just always use the parent configuration
    std::vector<PortConfig> inConfs;
    std::vector<Input::InputConfig> graphInputConfig;

    constexpr bool isInPlace = true;

    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto desc = getParentOutputMemDesc(getParentEdgeAt(i));
        inConfs.emplace_back(desc);
        graphInputConfig.emplace_back(node::Input::InputConfig{std::move(desc), isInPlace});
    }

    std::vector<Input::OutputConfig> graphOutputConfig(outputShapes.size(), node::Input::OutputConfig{true, isInPlace});

    // configure the inner graph to get the information about output memory descriptors
    m_graph.Init(m_body, context, graphInputConfig, graphOutputConfig);

    // for the output descriptors, use the configuration of the graph's output nodes
    auto outputDescriptors = m_graph.getOutputMemoryDescriptors();

    std::vector<PortConfig> outConfs;
    for (const auto& desc : outputDescriptors) {
        outConfs.emplace_back(desc);
    }

    const NodeConfig config(std::move(inConfs), std::move(outConfs));

    supportedPrimitiveDescriptors.clear();
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::undef);

    selectPrimitiveDescriptorByIndex(0);
}

// @todo add ascii diagramm for memory mapping / reuse
void Composite::createPrimitive() {
    m_graph.Activate();
}

int Composite::registerToAllocationContext(int offset, AllocationContext& context) {
    CPU_NODE_ASSERT(getOriginalInputsNumber() == m_graph.inputsNumber(),
                    "Number of node inputs must be equal the number of inner graph's inputs");

    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        auto inputEdges = m_graph.getInputNodeByIndex(i)->getChildEdgesAtPort(0);
        for (const auto& inputEdge : inputEdges) {
            CPU_NODE_ASSERT(inputEdge->getStatus() == Edge::Status::Uninitialized,
                            "Expected Uninitialized state for edge: ",
                            *this);
            inputEdge->sharedMemFrom(parentEdge);
        }
    }

    CPU_NODE_ASSERT(getOriginalOutputsNumber() == m_graph.outputsNumber(),
                    "Number of node outputs must be equal the number of inner graph's outputs");

    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        auto childEdge = getChildEdgeAt(i);
        auto outputEdge = m_graph.getOutputNodeByIndex(i)->getParentEdgeAt(0);
        CPU_NODE_ASSERT(outputEdge->getStatus() == Edge::Status::Uninitialized,
                        "Expected Uninitialized state for edge: ",
                        *outputEdge);
        outputEdge->sharedMemFrom(childEdge);
    }

    return m_graph.RegisterToAllocationContext(offset, context);
}

void Composite::execute(const dnnl::stream&) {
    m_graph.Infer();
}

void Composite::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);

    // since the shape inference is not performed for the composite node
    // a memory of the extra child edges, attached to the output ports
    // has to be updated after an inference of the inner graph finished
    auto& childEdges = getChildEdges();
    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        const auto mem = getDstMemoryAtPort(i);
        for (size_t j = getOriginalOutputsNumber(); j < childEdges.size(); j++) {
            auto& childEdge = childEdges[j];
            auto childEdgePtr = childEdge.lock();
            assert(childEdgePtr);

            if (childEdgePtr->getInputNum() == static_cast<int>(i)) {
                childEdgePtr->getMemoryPtr()->redefineDesc(mem->getDescPtr());
            }
        }
    }
}

}  // namespace ov::intel_cpu::node
