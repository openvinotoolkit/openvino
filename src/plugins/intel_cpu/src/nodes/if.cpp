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
#include "utils/general_utils.h"

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
      m_op(ov::as_type_ptr<ov::op::v8::If>(op)),
      bothSubGraphsAreNonConstant(!m_op->get_then_body()->get_parameters().empty() &&
                                  !m_op->get_else_body()->get_parameters().empty()) {
    CPU_NODE_ASSERT(m_op, "'If' operation is expected");

    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

void If::selectOptimalPrimitiveDescriptor() {
    // for the input configuration, just always use the parent configuration
    auto ifOp = ov::as_type_ptr<ov::op::v8::If>(m_op);
    const auto numThenParameters = ifOp->get_then_body()->get_parameters().size();
    const auto numThenResults = ifOp->get_then_body()->get_results().size();
    const auto numElseParameters = ifOp->get_else_body()->get_parameters().size();
    const auto numElseResults = ifOp->get_else_body()->get_results().size();

    std::vector<PortConfig> inConfs(inputShapes.size());
    std::vector<PortConfig> outConfs(outputShapes.size());

    std::vector<Input::InputConfig> thenInputConfig(numThenParameters);
    std::vector<Input::InputConfig> elseInputConfig(numElseParameters);

    const bool isInPlace = bothSubGraphsAreNonConstant;
    const bool elseInPlace = bothSubGraphsAreNonConstant;

    std::vector<Input::OutputConfig> thenOutputConfig(numThenResults, node::Input::OutputConfig{true, isInPlace});
    std::vector<Input::OutputConfig> elseOutputConfig(numElseResults, node::Input::OutputConfig{true, elseInPlace});

    auto thenInputDescriptions = ifOp->get_input_descriptions(0);
    auto elseInputDescriptions = ifOp->get_input_descriptions(1);

    auto conditionDesc = getParentOutputMemDesc(getParentEdgeAt(0));
    inConfs.at(0) = PortConfig(conditionDesc);

    for (const auto& description : thenInputDescriptions) {
        const auto inIdx = description->m_input_index;
        const auto paramIdx = description->m_body_parameter_index;
        auto desc = getParentOutputMemDesc(getParentEdgeAt(inIdx));
        inConfs.at(inIdx) = PortConfig(desc);
        thenInputConfig.at(paramIdx) = node::Input::InputConfig{desc, isInPlace};
    }

    for (const auto& description : elseInputDescriptions) {
        const auto inIdx = description->m_input_index;
        const auto paramIdx = description->m_body_parameter_index;
        auto desc = getParentOutputMemDesc(getParentEdgeAt(inIdx));
        inConfs.at(inIdx) = PortConfig(desc);
        elseInputConfig.at(paramIdx) = node::Input::InputConfig{desc, elseInPlace};
    }

    // configure the inner graph to get the information about output memory descriptors
    m_thenGraph.Init(ifOp->get_then_body(), context, thenInputConfig, thenOutputConfig);
    m_elseGraph.Init(ifOp->get_else_body(), context, elseInputConfig, elseOutputConfig);

    // for the output descriptors, use the configuration of the graph's output nodes
    auto thenOutputDescriptors = m_thenGraph.getOutputMemoryDescriptors();
    auto elseOutputDescriptors = m_elseGraph.getOutputMemoryDescriptors();
    auto thenOutputDescriptions = ifOp->get_output_descriptions(0);
    auto elseOutputDescriptions = ifOp->get_output_descriptions(1);

    for (const auto& description : thenOutputDescriptions) {
        auto outIdx = description->m_output_index;
        auto resultIdx = description->m_body_value_index;
        outConfs.at(outIdx) = PortConfig(thenOutputDescriptors.at(resultIdx));
    }

    for (const auto& description : elseOutputDescriptions) {
        auto outIdx = description->m_output_index;
        auto resultIdx = description->m_body_value_index;
        outConfs.at(outIdx) = PortConfig(elseOutputDescriptors.at(resultIdx));
    }

    const NodeConfig config(inConfs, outConfs);

    supportedPrimitiveDescriptors.clear();

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::undef);

    selectPrimitiveDescriptorByIndex(0);
}

int If::registerToAllocationContext(int offset, AllocationContext& context) {
    auto shareOuterGraphInputMem =
        [&](const Graph& graph,
            const op::util::MultiSubGraphOp::MultiSubgraphInputDescriptionVector& inputDescriptions) {
            for (const auto& description : inputDescriptions) {
                const auto inIdx = description->m_input_index;
                const auto paramIdx = description->m_body_parameter_index;
                auto parentEdge = getParentEdgeAt(inIdx);
                auto inputEdges = graph.getInputNodeByIndex(paramIdx)->getChildEdgesAtPort(0);
                for (const auto& inputEdge : inputEdges) {
                    OPENVINO_ASSERT(inputEdge->getStatus() == Edge::Status::Uninitialized,
                                    "Expected Uninitialized state for edge: ",
                                    *this);
                    std::cout << *inputEdge << " sharing memory from parent edge: " << *parentEdge << std::endl;
                    inputEdge->sharedMemFrom(parentEdge);
                }
            }
        };

    auto shareOuterGraphOutputMem =
        [&](const Graph& graph,
            const op::util::MultiSubGraphOp::MultiSubgraphOutputDescriptionVector& outputDescriptions) {
            for (const auto& description : outputDescriptions) {
                auto outIdx = description->m_output_index;
                auto resultIdx = description->m_body_value_index;
                std::cout << "Output description: " << description->m_output_index << " "
                          << description->m_body_value_index << std::endl;

                auto childEdge = getChildEdgeAt(outIdx);
                auto outputEdge = graph.getOutputNodeByIndex(resultIdx)->getParentEdgeAt(0);

                // input / output descriptions can contain duplicated entries
                if (outputEdge->getStatus() != Edge::Status::Uninitialized) {
                    continue;
                }

                std::cout << *outputEdge << " sharing memory from child edge: " << *childEdge << std::endl;
                outputEdge->sharedMemFrom(childEdge);
            }
        };
    // limit memory sharing to non-constant graphs for now
    // it is possible to share memory for constant graphs as well
    // but this would require extra back and forth memory manipulations
    if (bothSubGraphsAreNonConstant) {
        auto thenInputDescriptions = m_op->get_input_descriptions(0);
        auto elseInputDescriptions = m_op->get_input_descriptions(1);
        auto thenOutputDescriptions = m_op->get_output_descriptions(0);
        auto elseOutputDescriptions = m_op->get_output_descriptions(1);

        // take into account an offset of the both subgraphs
        shareOuterGraphInputMem(m_thenGraph, thenInputDescriptions);
        shareOuterGraphOutputMem(m_thenGraph, thenOutputDescriptions);
        shareOuterGraphInputMem(m_elseGraph, elseInputDescriptions);
        shareOuterGraphOutputMem(m_elseGraph, elseOutputDescriptions);
    }

    offset = m_thenGraph.RegisterToAllocationContext(offset, context);
    offset = m_elseGraph.RegisterToAllocationContext(offset, context);

    return offset;
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

    if (bothSubGraphsAreNonConstant) {
        return;  // no need to prepare mappers when memory is shared
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

void If::executeWithMemoryReuse(const dnnl::stream& strm) {
    const bool condition = static_cast<const bool>((getSrcDataAtPortAs<const uint8_t>(0))[0]);

    auto& subGraph = condition ? m_thenGraph : m_elseGraph;
    const auto& inputMap = condition ? thenInputPortMap : elseInputPortMap;
    // update input memory descriptors for the subgraphs
    for (const auto [ifPort, subGraphPort] : inputMap) {
        const auto& dstDesc = getSrcMemoryAtPort(ifPort)->getDescPtr();
        const auto& dstMem = subGraph.getInputNodeByIndex(subGraphPort)->getDstMemoryAtPort(0);
        dstMem->redefineDesc(dstDesc);
    }

    subGraph.ResetInferCount();
    subGraph.Infer();
}

void If::execute(const dnnl::stream& strm) {
    if (bothSubGraphsAreNonConstant) {
        return executeWithMemoryReuse(strm);
    }
    // if one of the subgraphs is constant, it does not participate in global memory reuse.
    // zero-copy switching between constant and non-constant memory requires extra processing.
    // fallback to the exlicit copy (mappers) approach for now
    const bool condition = static_cast<const bool>((getSrcDataAtPortAs<const uint8_t>(0))[0]);
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
