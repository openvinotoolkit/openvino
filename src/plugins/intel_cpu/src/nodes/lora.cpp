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

    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto desc = getParentOutputMemDesc(getParentEdgeAt(i));
        inConfs.emplace_back(desc);
        graphInputConfig.emplace_back(node::Input::InputConfig{desc, true});
    }

    std::vector<Input::OutputConfig> graphOutputConfig;
    // enforce the same memory descriptor on the output as on the input to allow inPlace memory
    graphOutputConfig.emplace_back(node::Input::OutputConfig{inConfs.front().getMemDesc(), true});

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

// @todo add ascii diagram for memory mapping / reuse
void LoRA::createPrimitive() {
    CPU_NODE_ASSERT(getOriginalInputsNumber() == m_graph.GetInputNodesMap().size(),
                    "Number of node inputs must be equal the number of inner graph's inputs");

    std::vector<MemoryPtr> inputMemory;
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        inputMemory.emplace_back(getSrcMemoryAtPort(i));
    }

    CPU_NODE_ASSERT(getOriginalOutputsNumber() == m_graph.GetOutputNodesMap().size(),
                    "Number of node outputs must be equal the number of inner graph's outputs");

    std::vector<MemoryPtr> outputMemory{getDstMemoryAtPort(0)};
    m_graph.Activate(inputMemory, outputMemory);
}

void LoRA::execute(dnnl::stream) {
    m_graph.Infer();
}

void LoRA::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
