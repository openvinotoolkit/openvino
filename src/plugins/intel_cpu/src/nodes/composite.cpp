// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "composite.h"

#include "cpu_memory.h"
#include "transformations/cpu_opset/common/op/submodel.hpp"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {
namespace node {

bool Composite::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    return ov::is_type<ov::intel_cpu::SubModel>(op);
}

Composite::Composite(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op, FULL_PORT_MASK)) {
    const auto& subModel = ov::as_type_ptr<SubModel>(op);
    OPENVINO_ASSERT(subModel, "Attempt to create SubGraph node from an invalid op type: ", op);

    m_body = subModel->get_function();
}

void Composite::selectOptimalPrimitiveDescriptor() {
    // for the input configution, just always use the parent configuration
    VecMemoryDescs inputDescriptors;
    for (size_t j = 0; j < getParentEdges().size(); j++) {
        inputDescriptors.emplace_back(getParentOutputMemDesc(getParentEdgeAt(0)));
    }

    std::vector<PortConfig> inConfs;
    for (const auto& desc : inputDescriptors) {
        inConfs.emplace_back(desc);
    }

    // configure the inner graph to get the information about output memory descriptors
    m_graph.Configure(m_body, context, inputDescriptors, true);

    // for the output decriptors, use the configuration of the graph's output nodes
    auto outputDescriptors = m_graph.getOutputMemoryDescriptors();

    std::vector<PortConfig> outConfs;
    for (const auto& desc : outputDescriptors) {
        outConfs.emplace_back(desc);
    }

    const NodeConfig config(inConfs, outConfs);

    supportedPrimitiveDescriptors.clear();
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::undef);

    selectPrimitiveDescriptorByIndex(0);
}

// @todo add ascii diagramm for memory mapping / reuse
void Composite::createPrimitive() {
    // Point a memory of the inner graph's input edges to the corresponding memory of the node parent edges
    OPENVINO_ASSERT(getOriginalInputsNumber() == m_graph.GetInputNodesMap().size(),
                    "Number of node inputs must be equal the number of inner graph's inputs");

    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        const auto input = m_graph.GetInputNodesMap()[i];

        for (size_t j = 0; j < input->getChildEdges().size(); j++) {
            input->getChildEdgeAt(j)->reuse(getSrcMemoryAtPort(i));
        }
    }

    // Point a memory of the inner graph's output edges to the corresponding memory of the node child edges
    // The extra child edges on output ports will be updated after the inference of the inner graph
    OPENVINO_ASSERT(getOriginalOutputsNumber() == m_graph.GetOutputNodesMap().size(),
                    "Number of node inputs must be equal the number of inner graph's inputs");

    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        const auto output = m_graph.GetOutputNodesMap()[i];
        output->getParentEdgeAt(0)->reuse(getDstMemoryAtPort(i));
    }

    // Allocate inner graph's memory
    m_graph.Allocate();
}

void Composite::execute(dnnl::stream) {
    m_graph.Infer();

    if (!inputShapesModified())
        return;

    // since the shape inference is not performed for the composite node
    // a memory of the extra child edges, attached to the output ports
    // has to be updated after an inference of the inner graph finished
    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        const auto mem = getDstMemoryAtPort(i);
        auto& childEdges = getChildEdges();
        for (size_t j = getOriginalOutputsNumber(); j < childEdges.size(); j++) {
            auto& childEdge = childEdges[j];
            auto childEdgePtr = childEdge.lock();
            if (childEdgePtr->getInputNum() == static_cast<int>(i)) {
                childEdgePtr->getMemoryPtr()->redefineDesc(mem->getDescPtr());
            }
        }
    }
}

void Composite::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
