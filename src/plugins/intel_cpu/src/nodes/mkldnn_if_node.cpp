// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_if_node.h"

#include "mkldnn_extension_utils.h"
#include "ie_ngraph_utils.hpp"
#include "transformations/utils/utils.hpp"
#include "common/memory_desc_wrapper.hpp"

#include <map>
#include <string>
#include <vector>

using namespace MKLDNNPlugin;


MKLDNNIfNode::PortMapHelper::PortMapHelper(const MKLDNNMemoryPtr &from, const MKLDNNMemoryPtr &to, const mkldnn::engine& eng) {
    // we should redefine 'to' memory for dynamism
    const auto &currDesc = to->getDesc();
    if (currDesc.getShape().isDynamic() || currDesc.getShape().getStaticDims() != from->getStaticDims()) {
        to->redefineDesc(from->getDesc());
    }

    mem_holder_src = from->GetPrimitive();
    mem_holder_dst = to->GetPrimitive();

    if (mem_holder_src.get_desc().get_size() != mem_holder_dst.get_desc().get_size())
        IE_THROW() << "PortMapHelper of 'If' has incorrect input and output memories for copying!";
    if (mem_holder_src.get_desc().data_type() != mem_holder_dst.get_desc().data_type())
        IE_THROW() << "PortMapHelper of 'If' has different data types of input and output memories for copying!";

    size = mem_holder_src.get_desc().get_size();
}

void MKLDNNIfNode::PortMapHelper::execute(mkldnn::stream& strm) {
    std::memcpy(get_ptr(mem_holder_dst), get_ptr(mem_holder_src), size);
}

void* MKLDNNIfNode::PortMapHelper::get_ptr(mkldnn::memory& prim) {
    auto ptr = static_cast<uint8_t*>(prim.get_data_handle());
    auto md = prim.get_desc().data;
    mkldnn::impl::memory_desc_wrapper wrapper(md);
    return ptr + wrapper.offset0() * wrapper.data_type_size();
}

bool MKLDNNIfNode::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(), ov::op::v8::If::get_type_info_static())) {
            errorMessage = "Not supported If operation version " + std::to_string(op->get_type_info().version) +
                    " with name '" + op->get_friendly_name() + "'. Node If supports only opset8 version.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNIfNode::MKLDNNIfNode(const std::shared_ptr<ov::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache), ovOp(op) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void MKLDNNIfNode::getSupportedDescriptors() {
    auto ifOp = ov::as_type_ptr<ov::op::v8::If>(ovOp);

    const std::shared_ptr<const ov::Model>& thenBody = ifOp->get_then_body();
    const std::shared_ptr<const ov::Model>& elseBody = ifOp->get_else_body();
    subGraphThen.CreateGraph(thenBody, ext_mng, weightCache);
    subGraphElse.CreateGraph(elseBody, ext_mng, weightCache);

    const auto &inMapThen = subGraphThen.GetInputNodesMap();
    for (const auto &param : ifOp->get_then_body()->get_parameters()) {
        auto inNode = inMapThen.find(param->get_friendly_name());
        if (inNode != inMapThen.end()) {
            auto inMem = inNode->second->getChildEdgeAt(0)->getMemoryPtr();
            inputMemThen.push_back(inMem);
        } else {
            IE_THROW() << "Then body of node If with name " << getName() << " does not have input with name: "
                    << param->get_friendly_name();
        }
    }

    const auto &inMapElse = subGraphElse.GetInputNodesMap();
    for (const auto &param : ifOp->get_else_body()->get_parameters()) {
        auto inNode = inMapElse.find(param->get_friendly_name());
        if (inNode != inMapElse.end()) {
            auto inMem = inNode->second->getChildEdgeAt(0)->getMemoryPtr();
            inputMemElse.push_back(inMem);
        } else {
            IE_THROW() << "Else body of node If with name " << getName() << " does not have input with name: "
                    << param->get_friendly_name();
        }
    }

    const auto &outMapThen = subGraphThen.GetOutputNodesMap();
    for (const auto& out : ifOp->get_then_body()->get_results()) {
        const auto prev = out->input_value(0);
        const std::string inputID = ngraph::op::util::get_ie_output_name(prev);
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
        const std::string inputID = ngraph::op::util::get_ie_output_name(prev);
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
}

void MKLDNNIfNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    NodeConfig config;
    config.inConfs.reserve(getParentEdges().size());
    config.outConfs.reserve(getChildEdges().size());

    for (size_t i = 0; i < inputShapes.size(); i++) {
        PortConfig dataConf {};
        auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        dataConf.desc = descCreator->createSharedDesc(getOriginalInputPrecisionAtPort(i), getInputShapeAtPort(i));
        config.inConfs.emplace_back(dataConf);
    }

    for (size_t i = 0; i < outputShapes.size(); i++) {
        PortConfig dataConf {};
        auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        dataConf.desc = descCreator->createSharedDesc(getOriginalOutputPrecisionAtPort(i), getOutputShapeAtPort(i));
        config.outConfs.push_back(dataConf);
    }

    config.dynBatchSupport = true;

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void MKLDNNIfNode::createPrimitive() {
    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

bool MKLDNNIfNode::needPrepareParams() const {
    return MKLDNNNode::inputShapesDefined() || condition != (reinterpret_cast<const uint8_t*>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr()))[0];
}

void MKLDNNIfNode::prepareParams() {
    const auto& eng = getEngine();

    if (isDynamic) {
        new_state = true;
        condition = (reinterpret_cast<const uint8_t*>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr()))[0];
        prepareBeforeMappers(condition, eng);
    } else {
        prepareBeforeMappers(true, eng);
        prepareBeforeMappers(false, eng);
        prepareAfterMappers(true, eng);
        prepareAfterMappers(false, eng);
    }
}

void MKLDNNIfNode::prepareBeforeMappers(const bool isThen, const dnnl::engine& eng) {
    auto &inputPortMap = isThen ? thenInputPortMap : elseInputPortMap;
    auto &inputMems = isThen ? inputMemThen : inputMemElse;
    auto &beforeMappers = isThen ? beforeThenMappers : beforeElseMappers;
    beforeMappers.clear();
    for (auto& map_rule : inputPortMap) {
        auto &fromMem = getParentEdgesAtPort(map_rule.from)[0]->getMemoryPtr();
        auto &toMem = inputMems[map_rule.to];

        beforeMappers.emplace_back(std::make_shared<PortMapHelper>(fromMem, toMem, eng));
    }
}

void MKLDNNIfNode::prepareAfterMappers(const bool isThen, const dnnl::engine& eng) {
    auto &outputPortMap = isThen ? thenOutputPortMap : elseOutputPortMap;
    auto &outputMems = isThen ? outputMemThen : outputMemElse;
    auto &afterMappers = isThen ? afterThenMappers : afterElseMappers;
    afterMappers.clear();
    for (auto& map_rule : outputPortMap) {
        auto &toMem = getChildEdgesAtPort(map_rule.from)[0]->getMemoryPtr();
        auto &fromMem = outputMems[map_rule.to];

        afterMappers.emplace_back(std::make_shared<PortMapHelper>(fromMem, toMem, eng));
    }
}

void MKLDNNIfNode::execute(mkldnn::stream strm) {
    if (!isDynamic)
        condition = (reinterpret_cast<const uint8_t*>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr()))[0];

    auto& beforeMappers = condition ? beforeThenMappers : beforeElseMappers;
    auto& afterMappers = condition ? afterThenMappers : afterElseMappers;
    auto& subGraph = condition ? subGraphThen : subGraphElse;

    for (auto &mapper : beforeMappers)
        mapper->execute(strm);
    subGraph.ResetInferCount();
    subGraph.Infer();

    // if condition was changed or there are new input shapes,
    // after subgraph inference we should redefine out memory of 'If'
    if (new_state) {
        new_state = false;
        const auto& eng = getEngine();
        prepareAfterMappers(condition, eng);
    }

    for (auto &mapper : afterMappers)
        mapper->execute(strm);
}

void MKLDNNIfNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

bool MKLDNNIfNode::created() const {
    return getType() == If;
}

REG_MKLDNN_PRIM_FOR(MKLDNNIfNode, If);
