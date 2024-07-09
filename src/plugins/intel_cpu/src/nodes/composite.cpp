// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "composite.h"
#include <future>

#include "cpu_memory.h"
#include "partitioned_mem_mgr.h"
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
    m_subStreamId = std::getenv("DISABLE_ASYNC") ? -1 : op->get_rt_info()["sub_stream_id"].as<int>();

    m_body = subModel->get_function();
}

void Composite::selectOptimalPrimitiveDescriptor() {
    // for the input configution, just always use the parent configuration
    VecMemoryDescs inputDescriptors;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        inputDescriptors.emplace_back(getParentOutputMemDesc(getParentEdgeAt(i)));
    }

    std::vector<PortConfig> inConfs;
    for (const auto& desc : inputDescriptors) {
        inConfs.emplace_back(desc);
    }

    // configure the inner graph to get the information about output memory descriptors
    auto newContext = std::make_shared<GraphContext>(
        context->getConfig(),
        context->getWeightsCaches(),
        context->isGraphQuantized(),
        context->getParamsCaches(),
        context->getCPUStreamExecutor(),
        m_subStreamId,
        context->getMemoryStatesRegister());

    m_graph.Configure(m_body, newContext, inputDescriptors, true, 1);

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

    // for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
    //     const auto input = m_graph.GetInputNodesMap()[i];

    //     for (size_t j = 0; j < input->getChildEdges().size(); j++) {
    //         // std::cout << "Input edge: " << *input->getChildEdgeAt(j) << " reuse parent edge: " << *getParentEdgeAt(i) << "\n";
    //         input->getChildEdgeAt(j)->reuse(getSrcMemoryAtPort(i));
    //     }
    // }

    // Point a memory of the inner graph's output edges to the corresponding memory of the node child edges
    // The extra child edges on output ports will be updated after the inference of the inner graph
    OPENVINO_ASSERT(getOriginalOutputsNumber() == m_graph.GetOutputNodesMap().size(),
                    "Number of node inputs must be equal the number of inner graph's inputs");

    // if (!std::getenv("USE_LOAD")) {
    //     for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
    //         const auto output = m_graph.GetOutputNodesMap()[i];
    //         std::cout << "Output edge: " << *output->getParentEdgeAt(0) << " reuse child edge: " << *getChildEdgeAt(i)
    //                   << "\n";
    //         // auto inputNum = output->getParentEdgeAt(0)->getInputNum();
    //         // auto parent = output->getParentEdgeAt(0)->getParent();
    //         // auto parentChildEdges = parent->getChildEdgesAtPort(inputNum);

    //         // for (auto& edge : parentChildEdges) {
    //         //     edge->reuse(getDstMemoryAtPort(0));
    //         // }
    //         output->getParentEdgeAt(0)->reuse(getDstMemoryAtPort(i));
    //     }
    // }
    std::vector<MemoryPtr> inputMemory;
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        // std::cout << "Composite input edge: " << *getParentEdgeAt(i) << "\n";
        inputMemory.emplace_back(getSrcMemoryAtPort(i));
    }

    std::vector<MemoryPtr> outputMemory;
    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        // std::cout << "Composite output edge: " << *getChildEdgeAt(i) << "\n";
        outputMemory.emplace_back(getDstMemoryAtPort(i));
    }

    // Allocate inner graph's memory
    auto streamExecutor = context->getCPUStreamExecutor();
    // Finish graph creating using appropriate substream (to have everything allocated numa-locally)
    if (m_subStreamId >= 0) {
        std::packaged_task<void()> task {
            [this, &inputMemory, &outputMemory](){
                m_graph.Allocate(inputMemory, outputMemory);
            }
        };
        auto future = task.get_future();
        streamExecutor->run_sub_stream(
            [&task](){
                task();
            },
            m_subStreamId);
        future.wait();
        future.get();
    } else {
        m_graph.Allocate(inputMemory, outputMemory);
    }

    // for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
    //     const auto output = m_graph.GetOutputNodesMap()[i];
    //     auto dstMemory = getDstMemoryAtPort(i);
    //     auto outputMemory = output->getSrcMemoryAtPort(0);
    //     auto outputMemoryMngr = outputMemory->getMemoryMngr();
    //     auto memMngr = std::make_shared<PartitionedMemoryMngr>(outputMemoryMngr);
    //     auto newMem = std::make_shared<Memory>(getEngine(), dstMemory->getDescPtr(), memMngr);
    //     getChildEdgeAt(i)->reuse(newMem);
    // }

    // for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
    //     const auto output = m_graph.GetOutputNodesMap()[i];
    //     getChildEdgeAt(i)->reuse(output->getSrcMemoryAtPort(0));
    // }

    // for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
    //     auto mem = getDstMemoryAtPort(i);
    //     for (size_t j = getOriginalOutputsNumber(); j < getChildEdges().size(); j++) {
    //         auto& childEdge = getChildEdges()[j];
    //         auto childEdgePtr  = childEdge.lock();
    //         if (childEdgePtr->getInputNum() == static_cast<int>(i)) {
    //             childEdgePtr->reuse(mem);
    //         }
    //     }
    // }
}

void Composite::execute(dnnl::stream) {
    // for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
    //     auto srcMem = getSrcMemoryAtPort(i);
    //     const auto input = m_graph.GetInputNodesMap()[i];

    //     for (size_t j = 0; j < input->getChildEdges().size(); j++) {
    //         input->getChildEdgeAt(j)->getMemory().getMemoryMngr()->setExtBuff(srcMem->getData(), srcMem->getSize());
    //     }
    // }

    m_graph.InferDynamicSync(nullptr);

    // for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
    //     auto dstMem = getDstMemoryAtPort(i);
    //     const auto output = m_graph.GetOutputNodesMap()[i];
    //     auto outputMem = output->getSrcMemoryAtPort(0);
    //     auto parentEdge = getParentEdgeAt(i);
    //     auto parent = parentEdge->getParent();
    //     auto inputPort = parentEdge->getInputNum();
    //     auto mainEdge = parent->getChildEdgeAt(inputPort);
    //     auto mainMem = mainEdge->getMemoryPtr();

    //     dstMem->getMemoryMngr()->setExtBuff(mainMem->getData(), mainMem->getSize());
    // }

    if (!inputShapesModified())
        return;

    // since the shape inference is not performed for the composite node
    // a memory of the extra child edges, attached to the output ports
    // has to be updated after an inference of the inner graph finished
    // for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
    //     auto edges = getChildEdgesAtPort(i);
    //     for (size_t j = 0; j < edges.size(); j++) {
    //         std::cout << "Child edge at port: " << i << " " << *edges[j] << " " << edges[j]->getMemory().getData() << "\n";
    //     }
    // }

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

    // auto& childEdges = getChildEdges();
    // for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
    //     auto mem = getDstMemoryAtPort(i);
    //     const auto output = m_graph.GetOutputNodesMap()[i];
    //     auto outputMem = output->getSrcMemoryAtPort(0);
    //     if (std::getenv("USE_LOAD")) {
    //         Node::redefineOutputMemory(i, outputMem->getStaticDims());
    //         mem->load(*outputMem);
    //     } else {
    //         for (size_t j = getOriginalOutputsNumber(); j < childEdges.size(); j++) {
    //             auto& childEdge = childEdges[j];
    //             auto childEdgePtr = childEdge.lock();
    //             assert(childEdgePtr);

    //             if (childEdgePtr->getInputNum() == static_cast<int>(i)) {
    //                 std::cout << "PostExecute, updating memory for edge: " << *childEdgePtr
    //                           << " from edge: " << *getChildEdgeAt(i) << " " << childEdgePtr->getMemory().getData()
    //                           << "\n";
    //                 childEdgePtr->getMemoryPtr()->redefineDesc(outputMem->getDescPtr());
    //                 std::cout << "PostExecute, updated memory for edge: " << *childEdgePtr
    //                           << " from edge: " << *getChildEdgeAt(i) << " " << childEdgePtr->getMemory().getData()
    //                           << "\n";
    //             }
    //         }
    //     }
    // }
}

void Composite::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
