// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "if.h"

#include <string>
#include <utility>
#include <vector>

#include "nodes/common/cpu_convert.h"
#include "nodes/node_config.h"
#include "openvino/core/except.hpp"
#include "openvino/op/if.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"

namespace ov::intel_cpu::node {

If::PortMapHelper::PortMapHelper(MemoryPtr from, std::deque<MemoryPtr> to, const dnnl::engine& eng)
    : srcMemPtr(std::move(from)),
      dstMemPtrs(std::move(to)),
      size(0) {
    if (srcMemPtr->getDesc().isDefined()) {
        size = srcMemPtr->getShape().getElementsCount();
    }

    // Backup dstMemPtrs
    for (auto& ptr : dstMemPtrs) {
        originalDstMemDescs.push_back(ptr->getDescPtr()->clone());
    }
}

void If::PortMapHelper::execute(const dnnl::stream& strm) {
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
    const auto& currDesc = dstMemPtrs.front()->getDesc();
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

If::If(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, InternalDynShapeInferFactory()),
      m_op(ov::as_type_ptr<ov::op::v8::If>(op)) {
    CPU_NODE_ASSERT(m_op, "'If' operation is expected");

    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

void If::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    m_thenGraph.Init(m_op->get_then_body(), context);
    m_elseGraph.Init(m_op->get_else_body(), context);

    NodeConfig config;
    config.inConfs.reserve(getParentEdges().size());
    config.outConfs.reserve(getChildEdges().size());

    for (size_t i = 0; i < inputShapes.size(); i++) {
        PortConfig dataConf{};
        auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        dataConf.setMemDesc(descCreator->createSharedDesc(getOriginalInputPrecisionAtPort(i), getInputShapeAtPort(i)));
        config.inConfs.emplace_back(dataConf);
    }

    for (size_t i = 0; i < outputShapes.size(); i++) {
        PortConfig dataConf{};
        auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        dataConf.setMemDesc(
            descCreator->createSharedDesc(getOriginalOutputPrecisionAtPort(i), getOutputShapeAtPort(i)));
        config.outConfs.push_back(dataConf);
    }

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

int If::registerToAllocationContext(int offset, AllocationContext& context) {
    // take into account an offset of the both subgraphs
    const int thenOffset = m_thenGraph.RegisterToAllocationContext(offset, context);
    const int elseOffset = m_elseGraph.RegisterToAllocationContext(thenOffset, context);
    return elseOffset;
}

void If::createPrimitive() {
    m_thenGraph.Activate();
    m_elseGraph.Activate();

    for (const auto& param : m_op->get_then_body()->get_parameters()) {
        if (auto inNode = m_thenGraph.getInputNodeByIndex(m_op->get_then_body()->get_parameter_index(param))) {
            inputMemThen.push_back(getToMemories(inNode.get(), 0));
        } else {
            THROW_CPU_NODE_ERR("Then body of node does not have input with name: ", param->get_friendly_name());
        }
    }

    for (const auto& param : m_op->get_else_body()->get_parameters()) {
        if (auto inNode = m_elseGraph.getInputNodeByIndex(m_op->get_else_body()->get_parameter_index(param))) {
            inputMemElse.push_back(getToMemories(inNode.get(), 0));
        } else {
            THROW_CPU_NODE_ERR("Else body of node does not have input with name: ", param->get_friendly_name());
        }
    }

    for (const auto& out : m_op->get_then_body()->get_results()) {
        if (auto outNode = m_thenGraph.getOutputNodeByIndex(m_op->get_then_body()->get_result_index(out))) {
            auto outMem = outNode->getSrcMemoryAtPort(0);
            outputMemThen.push_back(outMem);
        } else {
            THROW_CPU_NODE_ERR("Then body of node does not have output with name: ", out->get_friendly_name());
        }
    }

    for (const auto& out : m_op->get_else_body()->get_results()) {
        if (auto outNode = m_elseGraph.getOutputNodeByIndex(m_op->get_else_body()->get_result_index(out))) {
            auto outMem = outNode->getSrcMemoryAtPort(0);
            outputMemElse.push_back(outMem);
        } else {
            THROW_CPU_NODE_ERR("Else body of node does not have output with name: ", out->get_friendly_name());
        }
    }

    // Port map: outputs
    for (const auto& desc : m_op->get_output_descriptions(0)) {
        auto body_output_idx = desc->m_body_value_index;
        thenOutputPortMap.emplace_back(
            PortMap{static_cast<int>(desc->m_output_index), static_cast<int>(body_output_idx)});
    }
    for (const auto& desc : m_op->get_output_descriptions(1)) {
        auto body_output_idx = desc->m_body_value_index;
        elseOutputPortMap.emplace_back(
            PortMap{static_cast<int>(desc->m_output_index), static_cast<int>(body_output_idx)});
    }

    for (const auto& desc : m_op->get_input_descriptions(0)) {
        auto body_input_index = desc->m_body_parameter_index;
        thenInputPortMap.emplace_back(
            PortMap{static_cast<int>(desc->m_input_index), static_cast<int>(body_input_index)});
    }
    for (const auto& desc : m_op->get_input_descriptions(1)) {
        auto body_input_index = desc->m_body_parameter_index;
        elseInputPortMap.emplace_back(
            PortMap{static_cast<int>(desc->m_input_index), static_cast<int>(body_input_index)});
    }

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
    auto& inputPortMap = isThen ? thenInputPortMap : elseInputPortMap;
    auto& inputMems = isThen ? inputMemThen : inputMemElse;
    auto& beforeMappers = isThen ? beforeThenMappers : beforeElseMappers;
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
    auto& outputPortMap = isThen ? thenOutputPortMap : elseOutputPortMap;
    auto& outputMems = isThen ? outputMemThen : outputMemElse;
    auto& afterMappers = isThen ? afterThenMappers : afterElseMappers;
    for (auto& map_rule : outputPortMap) {
        auto toMems = getToMemories(this, map_rule.from);
        auto& fromMem = outputMems[map_rule.to];
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
    for (const auto& edge : node->getChildEdgesAtPort(port)) {
        memories.push_back(edge->getMemoryPtr());
    }
    return memories;
}

void If::execute(const dnnl::stream& strm) {
    const auto condition = static_cast<const bool>((getSrcDataAtPortAs<const uint8_t>(0))[0]);

    auto& beforeMappers = condition ? beforeThenMappers : beforeElseMappers;
    auto& afterMappers = condition ? afterThenMappers : afterElseMappers;
    auto& subGraph = condition ? m_thenGraph : m_elseGraph;

    for (auto& mapper : beforeMappers) {
        mapper->execute(strm);
    }

    subGraph.ResetInferCount();
    subGraph.Infer();

    for (auto& mapper : afterMappers) {
        mapper->execute(strm);
    }
}

void If::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool If::created() const {
    return getType() == Type::If;
}

}  // namespace ov::intel_cpu::node
