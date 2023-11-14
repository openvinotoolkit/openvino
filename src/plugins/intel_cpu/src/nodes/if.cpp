// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "if.h"

#include <dnnl_extension_utils.h>
#include "ie_ngraph_utils.hpp"
#include "transformations/utils/utils.hpp"
#include "common/cpu_memcpy.h"
#include <shape_inference/shape_inference_internal_dyn.hpp>

#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

If::PortMapHelper::PortMapHelper(const MemoryPtr &from, const std::deque<MemoryPtr>& to,
                                           const dnnl::engine& eng, int from_port, int to_port) : srcMemPtr(from), dstMemPtrs(to), src_port(from_port), dst_port(to_port) {
    // size = 0;
    // if (srcMemPtr->getDesc().isDefined())
    //     size = srcMemPtr->getSize();
}

void If::PortMapHelper::update(ProxyMemoryMngrPtr dstMemMngr) {
    auto srcDesc = srcMemPtr->getDescPtr();

    // FIXME: do we need all dstMemDesc compatible?
    auto dstDesc = dstMemPtrs.front()->getDescPtr();
    if (dstDesc->isCompatible(*srcDesc)) {
        dstMemMngr->setMemMngr(srcMemPtr->getMemoryMngr());  // TODO: setMemMngrResize
    } else {
        dstMemMngr->reset();
    }
}

void If::PortMapHelper::execute(dnnl::stream& strm) {
    // if output shapes are changed,
    // after subgraph inference we should redefine out memory of 'If'
    // redefineTo();

    // cpu_memcpy(dstMemPtrs.front()->getData(), srcMemPtr->getData(), size);

    if (dstMemPtrs.front()->getData() != srcMemPtr->getData()) {
        redefineTo();
        dstMemPtrs.front()->load(*srcMemPtr);
    }
}

void If::PortMapHelper::redefineTo() {
    const auto &currDesc = dstMemPtrs.front()->getDesc();
    if (currDesc.getShape().isDynamic() || currDesc.getShape().getStaticDims() != srcMemPtr->getStaticDims()) {
        // TODO : check the entire dstMemPtrs usage considering the proper memory sharing
        auto memDesc = srcMemPtr->getDescPtr();
        for (size_t j = 0; j < dstMemPtrs.size(); j++) {
            dstMemPtrs[j]->redefineDesc(memDesc);
        }

        // size = srcMemPtr->getSize();
    }
}

bool If::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(), ov::op::v8::If::get_type_info_static())) {
            errorMessage = "Not supported If operation version " + std::string(op->get_type_info().version_id) +
                    " with name '" + op->get_friendly_name() + "'. Node If supports only opset8 version.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

If::If(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context) :
        Node(op, context, InternalDynShapeInferFactory()), ovOp(op) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void If::getSupportedDescriptors() {
    auto ifOp = ov::as_type_ptr<ov::op::v8::If>(ovOp);

    const std::shared_ptr<const ov::Model>& thenBody = ifOp->get_then_body();
    const std::shared_ptr<const ov::Model>& elseBody = ifOp->get_else_body();
    subGraphThen.IsSubgraphOf(this);
    subGraphElse.IsSubgraphOf(this);
    subGraphThen.CreateGraph(thenBody, context);
    subGraphElse.CreateGraph(elseBody, context);

    const auto &inMapThen = subGraphThen.GetInputNodesMap();
    for (const auto &param : ifOp->get_then_body()->get_parameters()) {
        auto inNode = inMapThen.find(param->get_friendly_name());
        if (inNode != inMapThen.end()) {
            inputMemThen.push_back(getToMemories(inNode->second.get(), 0));
        } else {
            IE_THROW() << "Then body of node If with name " << getName() << " does not have input with name: "
                    << param->get_friendly_name();
        }
    }

    const auto &inMapElse = subGraphElse.GetInputNodesMap();
    for (const auto &param : ifOp->get_else_body()->get_parameters()) {
        auto inNode = inMapElse.find(param->get_friendly_name());
        if (inNode != inMapElse.end()) {
            inputMemElse.push_back(getToMemories(inNode->second.get(), 0));
        } else {
            IE_THROW() << "Else body of node If with name " << getName() << " does not have input with name: "
                    << param->get_friendly_name();
        }
    }

    const auto &outMapThen = subGraphThen.GetOutputNodesMap();
    for (const auto& out : ifOp->get_then_body()->get_results()) {
        const auto prev = out->input_value(0);
        const std::string inputID = ov::op::util::get_ie_output_name(prev);
        auto outNode = outMapThen.find(inputID);
        if (outNode != outMapThen.end()) {
            auto outMem = outNode->second->getParentEdgeAt(0)->getMemoryPtr();
            outputMemThen.push_back(outMem);
        } else {
            IE_THROW() << "Then body of node If with name " << getName() << " does not have output with name: "
                    << inputID;
        }
    }

    const auto &outMapElse = subGraphElse.GetOutputNodesMap();
    for (const auto& out : ifOp->get_else_body()->get_results()) {
        const auto prev = out->input_value(0);
        const std::string inputID = ov::op::util::get_ie_output_name(prev);
        auto outNode = outMapElse.find(inputID);
        if (outNode != outMapElse.end()) {
            auto outMem = outNode->second->getParentEdgeAt(0)->getMemoryPtr();
            outputMemElse.push_back(outMem);
        } else {
            IE_THROW() << "Else body of node If with name " << getName() << " does not have output with name: "
                    << inputID;
        }
    }

    // Port map: outputs
    for (const auto& desc : ifOp->get_output_descriptions(0)) {
        auto body_output_idx = desc->m_body_value_index;
        thenOutputPortMap.emplace_back(PortMap {
            static_cast<int>(desc->m_output_index), static_cast<int>(body_output_idx)});
    }
    for (const auto& desc : ifOp->get_output_descriptions(1)) {
        auto body_output_idx = desc->m_body_value_index;
        elseOutputPortMap.emplace_back(PortMap {
            static_cast<int>(desc->m_output_index), static_cast<int>(body_output_idx)});
    }

    for (const auto& desc : ifOp->get_input_descriptions(0)) {
        auto body_input_index = desc->m_body_parameter_index;
        thenInputPortMap.emplace_back(PortMap {
            static_cast<int>(desc->m_input_index), static_cast<int>(body_input_index)});
    }
    for (const auto& desc : ifOp->get_input_descriptions(1)) {
        auto body_input_index = desc->m_body_parameter_index;
        elseInputPortMap.emplace_back(PortMap {
            static_cast<int>(desc->m_input_index), static_cast<int>(body_input_index)});
    }

    //
    const auto &paramsThen = ifOp->get_then_body()->get_parameters();
    for (int i = 0; i < paramsThen.size(); i++) {
        auto inNode = subGraphThen.inputNodesMemMngrMap.find(paramsThen[i]->get_friendly_name());
        if (inNode != subGraphThen.inputNodesMemMngrMap.end()) {
            inputThenMemMngrs[i] = inNode->second;
        } else {
            IE_THROW() << "Then body of node If with name " << getName() << " does not have proxy memory manager for input with name: "
                    << paramsThen[i]->get_friendly_name();
        }
    }
    const auto &paramsElse = ifOp->get_else_body()->get_parameters();
    for (int i = 0; i < paramsElse.size(); i++) {
        auto inNode = subGraphElse.inputNodesMemMngrMap.find(paramsElse[i]->get_friendly_name());
        if (inNode != subGraphElse.inputNodesMemMngrMap.end()) {
            inputElseMemMngrs[i] = inNode->second;
        } else {
            IE_THROW() << "Else body of node If with name " << getName() << " does not have proxy memory manager for input with name: "
                    << paramsElse[i]->get_friendly_name();
        }
    }
}

void If::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    NodeConfig config;
    config.inConfs.reserve(getParentEdges().size());
    config.outConfs.reserve(getChildEdges().size());

    for (size_t i = 0; i < inputShapes.size(); i++) {
        PortConfig dataConf {};
        auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        dataConf.setMemDesc(descCreator->createSharedDesc(getOriginalInputPrecisionAtPort(i), getInputShapeAtPort(i)));
        config.inConfs.emplace_back(dataConf);
    }

    for (size_t i = 0; i < outputShapes.size(); i++) {
        PortConfig dataConf {};
        auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        dataConf.setMemDesc(descCreator->createSharedDesc(getOriginalOutputPrecisionAtPort(i), getOutputShapeAtPort(i)));
        config.outConfs.push_back(dataConf);
    }

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void If::createPrimitive() {
    const auto& eng = getEngine();
    prepareBeforeMappers(true, eng);
    prepareBeforeMappers(false, eng);
    prepareAfterMappers(true, eng);
    prepareAfterMappers(false, eng);

    if (inputShapesDefined()) {
        updateLastInputDims();
    }
}

void If::prepareBeforeMappers(const bool isThen, const dnnl::engine& eng) {
    auto &inputPortMap = isThen ? thenInputPortMap : elseInputPortMap;
    auto &inputMems = isThen ? inputMemThen : inputMemElse;
    auto &beforeMappers = isThen ? beforeThenMappers : beforeElseMappers;
    for (auto& map_rule : inputPortMap) {
        auto fromMem = getParentEdgesAtPort(map_rule.from)[0]->getMemoryPtr();
        auto &toMems = inputMems[map_rule.to];

        beforeMappers.emplace_back(std::make_shared<PortMapHelper>(fromMem, toMems, eng, map_rule.from, map_rule.to));
    }
}

void If::prepareAfterMappers(const bool isThen, const dnnl::engine& eng) {
    auto &outputPortMap = isThen ? thenOutputPortMap : elseOutputPortMap;
    auto &outputMems = isThen ? outputMemThen : outputMemElse;
    auto &afterMappers = isThen ? afterThenMappers : afterElseMappers;
    for (auto& map_rule : outputPortMap) {
        auto toMems = getToMemories(this, map_rule.from);
        auto &fromMem = outputMems[map_rule.to];

        afterMappers.emplace_back(std::make_shared<PortMapHelper>(fromMem, toMems, eng, map_rule.to, map_rule.from));
    }
}

std::deque<MemoryPtr> If::getToMemories(const Node* node, const size_t port) const {
    std::deque<MemoryPtr> memories;
    for (auto edge : node->getChildEdgesAtPort(port))
        memories.push_back(edge->getMemoryPtr());
    return memories;
}

void If::execute(dnnl::stream strm) {
    const bool condition = static_cast<const bool>((reinterpret_cast<const uint8_t*>(getParentEdgeAt(0)->getMemoryPtr()->getData()))[0]);

    auto& beforeMappers = condition ? beforeThenMappers : beforeElseMappers;
    auto& afterMappers = condition ? afterThenMappers : afterElseMappers;
    auto& subGraph = condition ? subGraphThen : subGraphElse;
    auto& inputMemMngrs = condition ? inputThenMemMngrs : inputElseMemMngrs;

    for (auto &mapper : beforeMappers) {
        mapper->update(inputMemMngrs[mapper->dst_port]);
        mapper->execute(strm);
    }
    for (auto &mapper : afterMappers)
        mapper->update(outputMemMngrs[mapper->dst_port]);
    subGraph.ResetInferCount();
    subGraph.Infer();
    for (auto &mapper : afterMappers)
        mapper->execute(strm);
}

void If::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool If::created() const {
    return getType() == Type::If;
}

void If::resolveInPlaceEdges(Edge::LOOK look) {
    if (!(look & Edge::LOOK_UP)) {
        Node::resolveInPlaceEdges(look);
        return;
    }

    auto selected_pd = getSelectedPrimitiveDescriptor();
    OPENVINO_ASSERT(selected_pd,
        "If ",
        getName(),
        " failed getSelectedPrimitiveDescriptor() call, preferable primitive descriptor is not set");

    for (auto& map_rule : thenOutputPortMap) {
        const auto output_index = map_rule.from;

        auto memDesc = selected_pd->getConfig().outConfs.at(output_index).getMemDesc();
        const auto memMngr = std::make_shared<ProxyMemoryMngr>();

        outputMemMngrs[map_rule.from] = memMngr;

        for (auto&& edge : getChildEdgesAtPort(output_index)) {
            OPENVINO_ASSERT(one_of(edge->getStatus(), Edge::Status::Uninitialized, Edge::Status::NotAllocated),
                " Unexpected inplace resolve call to an allocated edge: ", edge->name());

            auto edgeMem = std::make_shared<Memory>(getEngine(), memDesc, memMngr);
            edge->reuse(edgeMem);
        }
    }
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
