// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "composite.h"

#include "nodes/input.h"
#include "cpu_memory.h"
#include "transformations/cpu_opset/common/op/submodel.hpp"
#include "utils/debug_capabilities.h"
#include "shape_inference/shape_inference_internal_dyn.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

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
    OPENVINO_ASSERT(subModel, "Attempt to create SubGraph node from an invalid op type: ", op);

    m_body = subModel->get_function();
}

void Composite::selectOptimalPrimitiveDescriptor() {
    // for the input configuration, just always use the parent configuration
    std::vector<PortConfig> inConfs;
    std::vector<Input::InputConfig> graphInputConfig;

    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto desc = getParentOutputMemDesc(getParentEdgeAt(i));
        inConfs.emplace_back(desc);
        graphInputConfig.emplace_back(node::Input::InputConfig{desc, true});
    }

    std::vector<Input::OutputConfig> graphOutputConfig;
    for (size_t i = 0; i < outputShapes.size(); i++) {
        graphOutputConfig.emplace_back(node::Input::OutputConfig{true, true});
    }

    // configure the inner graph to get the information about output memory descriptors
    m_graph.Init(m_body, context, graphInputConfig, graphOutputConfig);

    // for the output descriptors, use the configuration of the graph's output nodes
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
    OPENVINO_ASSERT(getOriginalInputsNumber() == m_graph.GetInputNodesMap().size(),
                    "Number of node inputs must be equal the number of inner graph's inputs");

    std::vector<MemoryPtr> inputMemory;
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        inputMemory.emplace_back(getSrcMemoryAtPort(i));
    }

    OPENVINO_ASSERT(getOriginalOutputsNumber() == m_graph.GetOutputNodesMap().size(),
                    "Number of node outputs must be equal the number of inner graph's outputs");

    std::vector<MemoryPtr> outputMemory;
    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        outputMemory.emplace_back(getDstMemoryAtPort(i));
    }

    m_graph.Activate(inputMemory, outputMemory);
}

void Composite::execute(dnnl::stream) {
    m_graph.Infer();
}

void Composite::executeDynamicImpl(dnnl::stream strm) {
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

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
