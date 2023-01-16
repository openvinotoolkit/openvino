// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "if.h"

#include "openvino/op/if.hpp"

#include "common/cpu_memcpy.h"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "nodes/common/cpu_convert.h"
#include "transformations/utils/utils.hpp"

#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

If::PortMapHelper::PortMapHelper(const MemoryPtr &from, const std::deque<MemoryPtr>& to,
                                           const dnnl::engine& eng) : srcMemPtr(from), dstMemPtrs(to) {
    size = 0;
    if (srcMemPtr->getDesc().isDefined())
        size = srcMemPtr->getShape().getElementsCount();

    // Backup dstMemPtrs
    for (auto& ptr : dstMemPtrs) {
        originalDstMemDescs.push_back(ptr->getDescPtr()->clone());
    }
}

void If::PortMapHelper::execute(dnnl::stream& strm) {
    // if output shapes are changed,
    // after subgraph inference we should redefine out memory of 'If'
    redefineTo();

    cpu_convert(srcMemPtr->getData(),
                dstMemPtrs.front()->getData(),
                srcMemPtr->getDesc().getPrecision(),
                dstMemPtrs.front()->getDesc().getPrecision(),
                size);
}

void If::PortMapHelper::redefineTo() {
    const auto &currDesc = dstMemPtrs.front()->getDesc();
    if (currDesc.getShape().isDynamic() || currDesc.getShape().getStaticDims() != srcMemPtr->getStaticDims()) {
        // TODO : check the entire dstMemPtrs usage considering the proper memory sharing
        auto newShape = srcMemPtr->getStaticDims();
        for (size_t j = 0; j < dstMemPtrs.size(); j++) {
            // Only the shape is updated, the memory type remains unchanged
            dstMemPtrs[j]->redefineDesc(originalDstMemDescs[j]->cloneWithNewDims(newShape));
        }

        size = srcMemPtr->getShape().getElementsCount();
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
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

void If::getSupportedDescriptors() {
    auto ifOp = ov::as_type_ptr<ov::op::v8::If>(ovOp);

    const std::shared_ptr<const ov::Model>& thenBody = ifOp->get_then_body();
    const std::shared_ptr<const ov::Model>& elseBody = ifOp->get_else_body();
    subGraphThen.CreateGraph(thenBody, context);
    subGraphElse.CreateGraph(elseBody, context);

    const auto& inMapThen = subGraphThen.GetInputNodesMap();
    for (const auto& param : ifOp->get_then_body()->get_parameters()) {
        auto inNode = inMapThen.find(ifOp->get_then_body()->get_parameter_index(param));
        if (inNode != inMapThen.end()) {
            inputMemThen.push_back(getToMemories(inNode->second.get(), 0));
        } else {
            OPENVINO_THROW("Then body of node If with name ",
                           getName(),
                           " does not have input with name: ",
                           param->get_friendly_name());
        }
    }

    const auto& inMapElse = subGraphElse.GetInputNodesMap();
    for (const auto& param : ifOp->get_else_body()->get_parameters()) {
        auto inNode = inMapElse.find(ifOp->get_else_body()->get_parameter_index(param));
        if (inNode != inMapElse.end()) {
            inputMemElse.push_back(getToMemories(inNode->second.get(), 0));
        } else {
            OPENVINO_THROW("Else body of node If with name ",
                           getName(),
                           " does not have input with name: ",
                           param->get_friendly_name());
        }
    }

    const auto &outMapThen = subGraphThen.GetOutputNodesMap();
    for (const auto& out : ifOp->get_then_body()->get_results()) {
        auto outNode = outMapThen.find(ifOp->get_then_body()->get_result_index(out));
        if (outNode != outMapThen.end()) {
            auto outMem = outNode->second->getSrcMemoryAtPort(0);
            outputMemThen.push_back(outMem);
        } else {
            OPENVINO_THROW("Then body of node If with name ", getName(), " does not have output with name: ", out->get_friendly_name());
        }
    }

    const auto &outMapElse = subGraphElse.GetOutputNodesMap();
    for (const auto& out : ifOp->get_else_body()->get_results()) {
        auto outNode = outMapElse.find(ifOp->get_else_body()->get_result_index(out));
        if (outNode != outMapElse.end()) {
            auto outMem = outNode->second->getSrcMemoryAtPort(0);
            outputMemElse.push_back(outMem);
        } else {
            OPENVINO_THROW("Else body of node If with name ", getName(), " does not have output with name: ", out->get_friendly_name());
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
        auto fromMem = getSrcMemoryAtPort(map_rule.from);
        auto& toMems = inputMems[map_rule.to];
        // Check precision between If node input/output and it's subgrapsh input/output.
        for (const auto& toMem : toMems) {
            if (fromMem->getDesc().getPrecision() != toMem->getDesc().getPrecision()) {
                DEBUG_LOG("If node fromMem and toMem precision mismatch: from ",
                          fromMem->getDesc().getPrecision().to_string(),
                          " to ",
                          toMem->getDesc().getPrecision().to_string());
            }
        }

        beforeMappers.emplace_back(std::make_shared<PortMapHelper>(fromMem, toMems, eng));
    }
}

void If::prepareAfterMappers(const bool isThen, const dnnl::engine& eng) {
    auto &outputPortMap = isThen ? thenOutputPortMap : elseOutputPortMap;
    auto &outputMems = isThen ? outputMemThen : outputMemElse;
    auto &afterMappers = isThen ? afterThenMappers : afterElseMappers;
    for (auto& map_rule : outputPortMap) {
        auto toMems = getToMemories(this, map_rule.from);
        auto &fromMem = outputMems[map_rule.to];
        // Check precision between If node input/output and it's subgrapsh input/output.
        for (const auto& toMem : toMems) {
            if (fromMem->getDesc().getPrecision() != toMem->getDesc().getPrecision()) {
                DEBUG_LOG("If node fromMem and toMem precision mismatch: from ",
                          fromMem->getDesc().getPrecision().to_string(),
                          " to ",
                          toMem->getDesc().getPrecision().to_string());
            }
        }

        afterMappers.emplace_back(std::make_shared<PortMapHelper>(fromMem, toMems, eng));
    }
}

std::deque<MemoryPtr> If::getToMemories(const Node* node, const size_t port) const {
    std::deque<MemoryPtr> memories;
    for (auto edge : node->getChildEdgesAtPort(port))
        memories.push_back(edge->getMemoryPtr());
    return memories;
}

void If::execute(dnnl::stream strm) {
    const bool condition = static_cast<const bool>((getSrcDataAtPortAs<const uint8_t>(0))[0]);

    auto& beforeMappers = condition ? beforeThenMappers : beforeElseMappers;
    auto& afterMappers = condition ? afterThenMappers : afterElseMappers;
    auto& subGraph = condition ? subGraphThen : subGraphElse;

    for (auto &mapper : beforeMappers)
        mapper->execute(strm);
    CPU_DEBUG_CAP_ENABLE(subGraph.ResetInferCount());
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

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
