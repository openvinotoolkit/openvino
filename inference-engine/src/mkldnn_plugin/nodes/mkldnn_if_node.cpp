// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_if_node.h"

#include <mkldnn_extension_utils.h>
#include <ie_ngraph_utils.hpp>

#include <map>
#include <string>
#include <vector>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine::details;

namespace MKLDNNPlugin {

static NodeConfig makePlainConfig(const std::shared_ptr<ngraph::Node>& op) {
    NodeConfig config;

    for (size_t i = 0; i < op->get_input_size(); i++) {
        auto dims = op->get_input_shape(i);
        if (dims.empty())
            dims = {1};
        const auto prec = InferenceEngine::details::convertPrecision(op->get_input_element_type(i));

        PortConfig data_conf {};
        auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        data_conf.desc = descCreator->createSharedDesc(prec, Shape(dims));
        config.inConfs.push_back(data_conf);
    }

    for (size_t i = 0; i < op->get_output_size(); i++) {
        auto dims = op->get_output_shape(i);
        if (dims.empty())
            dims = {1};
        const auto prec = InferenceEngine::details::convertPrecision(op->get_output_element_type(i));

        PortConfig data_conf {};
        auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        data_conf.desc = descCreator->createSharedDesc(prec, Shape(dims));
        config.outConfs.push_back(data_conf);
    }

    config.dynBatchSupport = true;
    return config;
}
}  // namespace MKLDNNPlugin

MKLDNNIfNode::PortMapHelper::PortMapHelper(const MKLDNNMemoryPtr &from, const MKLDNNMemoryPtr &to, const mkldnn::engine& eng) {
    mem_holder_src = from->GetPrimitive();
    mem_holder_dst = to->GetPrimitive();
    reorder = {mem_holder_src, mem_holder_dst};
}

void MKLDNNIfNode::PortMapHelper::execute(mkldnn::stream& strm) {
    reorder.execute(strm, mem_holder_src, mem_holder_dst);
}

bool MKLDNNIfNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                ngraph::op::v8::If::type_info)) {
            errorMessage = "Not supported If operation version with name " + op->get_friendly_name();
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNIfNode::MKLDNNIfNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache), ngraphOp(op) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void MKLDNNIfNode::getSupportedDescriptors() {
    auto ifOp = std::dynamic_pointer_cast<ngraph::op::v8::If>(ngraphOp);
    if (ifOp == nullptr) {
        IE_THROW() << "Invalid If operation type with name: " << getName();
    }
    const std::shared_ptr<const ngraph::Function>& thenBody = ifOp->get_then_body();
    const std::shared_ptr<const ngraph::Function>& elseBody = ifOp->get_else_body();
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
        auto prev = out->get_input_node_shared_ptr(0);
        std::string inputID = prev->get_friendly_name();
        if (prev->get_output_size() > 1) {
            inputID += "." + std::to_string(out->get_input_source_output(0).get_index());
        }
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
        auto prev = out->get_input_node_shared_ptr(0);
        std::string inputID = prev->get_friendly_name();
        if (prev->get_output_size() > 1) {
            inputID += "." + std::to_string(out->get_input_source_output(0).get_index());
        }
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

    config = makePlainConfig(ngraphOp);
}

void MKLDNNIfNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}


void MKLDNNIfNode::createPrimitive() {
    const auto& eng = getEngine();

    for (auto& map_rule : thenInputPortMap) {
        auto &fromMem = getParentEdgesAtPort(map_rule.from)[0]->getMemoryPtr();
        auto &toMem = inputMemThen[map_rule.to];

        beforeThenMappers.emplace_back(new PortMapHelper(fromMem, toMem, eng));
    }

    for (auto& map_rule : elseInputPortMap) {
        auto &fromMem = getParentEdgesAtPort(map_rule.from)[0]->getMemoryPtr();
        auto &toMem = inputMemElse[map_rule.to];

        beforeElseMappers.emplace_back(new PortMapHelper(fromMem, toMem, eng));
    }

    for (auto& map_rule : thenOutputPortMap) {
        auto &toMem = getChildEdgesAtPort(map_rule.from)[0]->getMemoryPtr();
        auto &fromMem = outputMemThen[map_rule.to];

        afterThenMappers.emplace_back(new PortMapHelper(fromMem, toMem, eng));
    }

    for (auto& map_rule : elseOutputPortMap) {
        auto &toMem = getChildEdgesAtPort(map_rule.from)[0]->getMemoryPtr();
        auto &fromMem = outputMemElse[map_rule.to];

        afterElseMappers.emplace_back(new PortMapHelper(fromMem, toMem, eng));
    }
}

void MKLDNNIfNode::execute(mkldnn::stream strm) {
    const bool condition = *(reinterpret_cast<const bool*>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr()));

    if (condition) {
        for (auto &mapper : beforeThenMappers)
            mapper->execute(strm);
        subGraphThen.ResetInferCount();
        subGraphThen.Infer();
        for (auto &mapper : afterThenMappers)
            mapper->execute(strm);
    } else {
        for (auto &mapper : beforeElseMappers)
            mapper->execute(strm);
        subGraphElse.ResetInferCount();
        subGraphElse.Infer();
        for (auto &mapper : afterElseMappers)
            mapper->execute(strm);
    }
}

bool MKLDNNIfNode::created() const {
    return getType() == If;
}

REG_MKLDNN_PRIM_FOR(MKLDNNIfNode, If);
