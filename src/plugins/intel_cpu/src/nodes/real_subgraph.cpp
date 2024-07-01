// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "real_subgraph.h"
#include "cpu_memory.h"
#include "transformations/cpu_opset/common/op/subgraph.hpp"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {
namespace node {

bool SubGraph::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    return ov::is_type<ov::intel_cpu::SubModel>(op);
}

SubGraph::SubGraph(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op, FULL_PORT_MASK)) {
    const auto& subModel = ov::as_type_ptr<SubModel>(op);
    m_subStreamToUse = std::getenv("DISABLE_ASYNC") ? -1 : op->get_rt_info()["sub_stream_num"].as<int>();

    OPENVINO_ASSERT(subModel, "Attempt to create SubGraph node from an invalid op type");
    m_body = subModel->get_function();
}

bool SubGraph::created() const { return true; }

void SubGraph::getSupportedDescriptors() {}

void SubGraph::selectOptimalPrimitiveDescriptor() {
    // for the input configution, just always use the parents configuration
    VecMemoryDescs inputDescriptors;
    for (size_t j = 0; j < getParentEdges().size(); j++) {
        auto parentEdge = getParentEdgeAt(j);
        auto parentPtr = parentEdge->getParent();
        auto parent_spd = parentPtr->getSelectedPrimitiveDescriptor();
        OPENVINO_ASSERT(parent_spd, "Parent selected primitive descriptor is missed");
        const auto& parentOutConf = parent_spd->getConfig().outConfs;
        OPENVINO_ASSERT(!parentOutConf.empty(), "Parent output configuration is empty");

        int inNum = parentEdge->getInputNum();
        auto parentDesc = parent_spd->getConfig().outConfs[inNum].getMemDesc();
        inputDescriptors.emplace_back(parentDesc);
    }

    std::vector<PortConfig> inConfs, outConfs;
    for (const auto& desc : inputDescriptors) {
        inConfs.emplace_back(desc);
    }

    auto streamExecutor = context->getCPUStreamExecutor();

    // Configure graph using appropriate substream (to have everything allocated numa-locally)
    if (m_subStreamToUse >= 0) {
        std::packaged_task<void()> task {
            [this, &inputDescriptors](){
                auto ctx = std::make_shared<GraphContext>(
                    context->getConfig(),
                    context->getWeightsCaches(),
                    context->isGraphQuantized(),
                    context->getParamsCaches(),
                    context->getCPUStreamExecutor(),
                    m_subStreamToUse,
                    context->getMemoryStatesRegister());
                m_graph.Configure(m_body, ctx, inputDescriptors, true, 1);
            }
        };
        auto future = task.get_future();
        streamExecutor->run_sub_stream(
            [&task](){
                task();
            },
            m_subStreamToUse);
        future.wait();
        future.get();
    } else {
        m_graph.Configure(m_body, context, inputDescriptors, true, 1);
    }

    // for output configuration, use the configuration of the graph's output node
    auto outputDescriptors = m_graph.getOutputMemoryDescriptors();

    for (const auto& desc : outputDescriptors) {
        outConfs.emplace_back(desc);
    }

    const NodeConfig config(inConfs, outConfs);
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::undef);
    selectPrimitiveDescriptorByIndex(0);
}

void SubGraph::prepareParams() {}

void SubGraph::infer() {
    m_graph.InferDynamicLightWell2();
}

void SubGraph::createPrimitive() {
    // for inputs it is straighforward, simply map memory on the input edges to the memory on subgraph's node parent edges
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        const auto input = m_graph.GetInputNodesMap()[i];

        for (size_t j = 0; j < input->getChildEdges().size(); j++) {
            input->getChildEdgeAt(j)->reuse(getSrcMemoryAtPort(i));
        }
    }

    // for outputs there are two options:
    // 1) do the same as for inputs (the code snippet commented out bellow)
    // 2) create graph first, and then perform the inverted mapping - map subgraph's node child edges to the memory from output child edges
    // Currently the second option is used (probably the first one should be prefered)

    // for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
    //     const auto output = m_graph.GetOutputNodesMap()[i];
    //     output->getParentEdgeAt(0)->reuse(getDstMemoryAtPort(i));
    // }

    // for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
    //     const auto output = m_graph.GetOutputNodesMap()[i];
    //     auto childEdges = getChildEdgesAtPort(i);
    //     for (auto& edge : childEdges) {
    //         edge->reuse(output->getSrcMemoryAtPort(0));
    //     }
    // }

    auto streamExecutor = context->getCPUStreamExecutor();
    // Finish graph creating using appropriate substream (to have everything allocated numa-locally)
    if (m_subStreamToUse >= 0) {
        std::packaged_task<void()> task {
            [this](){
                m_graph.Finish();
            }
        };
        auto future = task.get_future();
        streamExecutor->run_sub_stream(
            [&task](){
                task();
            },
            m_subStreamToUse);
        future.wait();
        future.get();
    } else {
        m_graph.Finish();
    }

    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        const auto output = m_graph.GetOutputNodesMap()[i];
        getChildEdgeAt(i)->reuse(output->getSrcMemoryAtPort(0));
    }

    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        auto mem = getDstMemoryAtPort(i);
        for (size_t j = getOriginalOutputsNumber(); j < getChildEdges().size(); j++) {
            auto& childEdge = getChildEdges()[j];
            auto childEdgePtr  = childEdge.lock();
            if (childEdgePtr->getInputNum() == i) {
                childEdgePtr->reuse(mem);
            }
        }
    }
}

void SubGraph::resolveInPlaceEdges(Edge::LOOK look) {}

void SubGraph::execute(dnnl::stream) {
    // simply infer the internal graph
    infer();
    // mark that infer is finished
    m_isRunning = false;
}

void SubGraph::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

} // namespace node
} // namespace intel_cpu
} // namespace ov
