// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_expert.h"

#include "cpu_memory.h"
#include "nodes/input.h"
#include "ov_ops/moe_expert.hpp"
#include "shape_inference/shape_inference_pass_through.hpp"
#include "utils/debug_capabilities.h"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu::node {

bool MOEExpert::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::op::internal::MOEExpert>(op)) {
            errorMessage = "Unknown MOEExpert operation : " + std::string(op->get_type_info().name) + " with name '" +
                           op->get_friendly_name() + "'";
        }
    } catch (...) {
        return false;
    }
    return true;
}

MOEExpert::MOEExpert(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, PassThroughShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto& MOEExpertModel = ov::as_type_ptr<ov::op::internal::MOEExpert>(op);
    CPU_NODE_ASSERT(MOEExpertModel,
                    "Attempt to create MOEExpert node from an invalid op type: ",
                    op,
                    " with name ",
                    op->get_friendly_name());

    m_body = MOEExpertModel->get_function();
    m_config = MOEExpertModel->get_config();
}

void MOEExpert::selectOptimalPrimitiveDescriptor() {
    // for the input configuration, just always use the parent configuration
    std::vector<PortConfig> inConfs;
    std::vector<Input::InputConfig> graphInputConfig;

    auto mainInputPrc = getOriginalInputPrecisionAtPort(0);

    constexpr bool isInPlace = true;
    {
        auto desc = getParentOutputMemDesc(getParentEdgeAt(0))->cloneWithNewPrecision(mainInputPrc);
        inConfs.emplace_back(desc);
        graphInputConfig.emplace_back(node::Input::InputConfig{desc, isInPlace});
    }

    // expert mask is i32
    auto expertMaskInputDesc = getParentOutputMemDesc(getParentEdgeAt(1));
    inConfs.emplace_back(expertMaskInputDesc);
    graphInputConfig.emplace_back(node::Input::InputConfig{expertMaskInputDesc, isInPlace});

    for (size_t i = 2; i < getParentEdges().size(); i++) {
        auto desc = getParentOutputMemDesc(getParentEdgeAt(i))->cloneWithNewPrecision(mainInputPrc);
        inConfs.emplace_back(desc);
        graphInputConfig.emplace_back(node::Input::InputConfig{desc, isInPlace});
    }

    std::vector<Input::OutputConfig> graphOutputConfig;
    // enforce the same memory descriptor on the output as on the input to allow inPlace memory
    graphOutputConfig.emplace_back(inConfs.front().getMemDesc(), false);

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

int MOEExpert::registerToAllocationContext(int offset, AllocationContext& context) {
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

void MOEExpert::createPrimitive() {
    CPU_NODE_ASSERT(getOriginalInputsNumber() == m_graph.inputsNumber(),
                    "Number of node inputs must be equal the number of inner graph's inputs");
    // Workaround to avoid making MOEExpert node always executable (isExecutable() = true)
    // This way we update subgraph's input memory without performing an actual Infer() call
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        auto subgraphInputNode = m_graph.getInputNodeByIndex(i);
        auto subgraphInputMemory = subgraphInputNode->getDstMemoryAtPort(0);
        subgraphMemoryPtrs.emplace_back(subgraphInputMemory);
    }

    m_graph.Activate();
}

void MOEExpert::execute(const dnnl::stream&) {
    // PlainTensor final_hidden_states;        // shape: [batch * seq_len, hidden_dim]
    PlainTensor expert_mask;                // shape: [expert_number, topk, batch]
    // PlainTensor hidden_states;              // shape: [1, batch * seq_len, hidden_dim]
    // PlainTensor routing_weights;            // shape: [self.topk * batch, 1]
    // PlainTensor dst;                        // shape: [batch * seq_len, hidden_dim]
    // final_hidden_states.reset(getSrcMemoryAtPort(0));
    expert_mask.reset(getSrcMemoryAtPort(1));
    // hidden_states.reset(getSrcMemoryAtPort(2));
    // routing_weights.reset(getSrcMemoryAtPort(3));
    // dst.reset(getDstMemoryAtPort(0));
    // std::cout << "final_hidden_states=" << final_hidden_states << "\n";
    // std::cout << "dst=" << dst << "\n";
    // std::cout << "expert_mask=" << expert_mask << "\n";
    // std::cout << "hidden_states=" << hidden_states << "\n";
    // std::cout << "routing_weights=" << routing_weights << "\n";
    bool flag = false;
    auto expert = expert_mask.ptr<int>(m_config.expert_no);
    for (size_t i = 0; i < expert_mask.m_dims[1] * expert_mask.m_dims[2]; i++) {
        if (expert[i]) {
            flag = true;
            break;
        }
    }
    if (flag) {
        m_graph.Infer();
    }
}

void MOEExpert::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void MOEExpert::prepareParams() {
    // final_hidden_states, expert_mask, hidden_states, routing_weights
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        // since the external and internal descriptors are compatible, we may pass the descriptor
        subgraphMemoryPtrs[i]->redefineDesc(getSrcMemoryAtPort(i)->getDescPtr());
    }
}

}  // namespace ov::intel_cpu::node
