// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "mkldnn_memory_node.hpp"
#include "common/cpu_memcpy.h"
#include "utils/general_utils.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

std::mutex MKLDNNMemoryNodeVirtualEdge::holderMutex;

MKLDNNMemoryNode::MKLDNNMemoryNode(const std::shared_ptr<ngraph::Node>& op) {
    if (auto assignOp = std::dynamic_pointer_cast<ngraph::op::AssignBase>(op)) {
        _id = assignOp->get_variable_id();
    } else if (auto readValueOp = std::dynamic_pointer_cast<ngraph::op::ReadValueBase>(op)) {
        _id = readValueOp->get_variable_id();
    }
}

bool MKLDNNMemoryOutputNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!MKLDNNPlugin::one_of(op->get_type_info(),
                ngraph::op::v3::Assign::type_info,
                ngraph::op::v6::Assign::type_info)) {
            errorMessage = "Node is not an instance of Assign from the operation set v3 or v6.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNMemoryOutputNode::MKLDNNMemoryOutputNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache) , MKLDNNMemoryNode(op) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    if (created()) {
        holder = MKLDNNMemoryNodeVirtualEdge::registerOutput(this);
    }
}

MKLDNNMemoryOutputNode::~MKLDNNMemoryOutputNode() {
    MKLDNNMemoryNodeVirtualEdge::remove(this, holder);
}

void MKLDNNMemoryOutputNode::getSupportedDescriptors() {}

void MKLDNNMemoryOutputNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getOriginalInputPrecisionAtPort(0);
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    NodeConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.inConfs[0].desc = make_unique<MKLDNNMemoryDesc>(getParentEdgeAt(0)->getShape().getStaticMklDims(), inputDataType,
                                                           MKLDNNMemory::GetPlainFormatByRank(getParentEdgeAt(0)->getShape().getRank()));
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void MKLDNNMemoryOutputNode::execute(mkldnn::stream strm)  {
    auto& srcMemory = getParentEdgeAt(0)->getMemory();

    auto inputMemoryNode = dynamic_cast<MKLDNNMemoryInputNode*>(inputNode);
    IE_ASSERT(inputMemoryNode != nullptr);
    inputMemoryNode->storeState(srcMemory);
}

bool MKLDNNMemoryInputNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!MKLDNNPlugin::one_of(op->get_type_info(),
                ngraph::op::v3::ReadValue::type_info,
                ngraph::op::v6::ReadValue::type_info)) {
            errorMessage = "Node is not an instance of ReadValue from the operation set v3 or v6.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNMemoryInputNode::MKLDNNMemoryInputNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNInputNode(op, eng, cache), MKLDNNMemoryNode(op), dataStore(new MKLDNNMemory{eng}) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    if (created()) {
        holder = MKLDNNMemoryNodeVirtualEdge::registerInput(this);
    }
}

void MKLDNNMemoryInputNode::createPrimitive() {
    MKLDNNInputNode::createPrimitive();

    dataStore->Create(getChildEdgeAt(0)->getMemory().GetDesc());

    // default memory state is zero filled
    dataStore->FillZero();
}

/**
 * Copy data from one tensor into other.
 * As is. Assume that data is dense tensor with same layout.
 * @param dst destination memory object
 * @param src source memory object
 */
inline
static void simple_copy(const MKLDNNMemory& dst, const MKLDNNMemory& src) {
    auto srcPtr = static_cast<uint8_t*>(src.GetPtr());
    auto dstPtr = static_cast<uint8_t*>(dst.GetPtr());
    auto srcSizeInByte = src.GetSize();
    auto dstSizeInByte = dst.GetSize();

    IE_ASSERT(srcSizeInByte == dstSizeInByte) << "Memory objects are not compatible. Has different sizes.";

    cpu_memcpy(dstPtr, srcPtr, srcSizeInByte);
}

MKLDNNMemoryInputNode::~MKLDNNMemoryInputNode() {
    MKLDNNMemoryNodeVirtualEdge::remove(this, holder);
}

MKLDNNMemoryPtr MKLDNNMemoryInputNode::getStore() {
    return dataStore;
}

void MKLDNNMemoryInputNode::storeState(const MKLDNNMemory &new_state) {
    // TODO: Should be next one call:
    //           dataStore.SetData(new_state, false);
    //       But because of performance reason we use simple manual copy
    simple_copy(*dataStore, new_state);
}

void MKLDNNMemoryInputNode::execute(mkldnn::stream strm) {
    // TODO: Should be simple call of:
    //           dst_mem.SetData(dataStore, false);
    //       But because of performance reason we use simple manual copy
    simple_copy(getChildEdgeAt(0)->getMemory(), *dataStore);
}

MKLDNNMemoryNodeVirtualEdge::Holder* MKLDNNMemoryNodeVirtualEdge::registerInput(MKLDNNMemoryInputNode * node) {
    std::lock_guard<std::mutex> lock{MKLDNNMemoryNodeVirtualEdge::holderMutex};
    // in case of output already registered
    auto& holder = MKLDNNMemoryNodeVirtualEdge::getExisted();
    auto sibling = MKLDNNMemoryNodeVirtualEdge::getByName(holder, node->getId());
    if (sibling != nullptr) {
        auto outputNode = dynamic_cast<MKLDNNMemoryOutputNode*>(sibling);
        IE_ASSERT(outputNode != nullptr);
        outputNode->setInputNode(node);
    } else {
        holder[node->getId()] = node;
    }
    return &holder;
}

MKLDNNMemoryNodeVirtualEdge::Holder* MKLDNNMemoryNodeVirtualEdge::registerOutput(MKLDNNMemoryOutputNode * node) {
    std::lock_guard<std::mutex> lock{MKLDNNMemoryNodeVirtualEdge::holderMutex};
    // in case of output layer
    auto& holder = MKLDNNMemoryNodeVirtualEdge::getExisted();
    auto sibling = MKLDNNMemoryNodeVirtualEdge::getByName(holder, node->getId());
    if (sibling != nullptr) {
        auto inputNode = dynamic_cast<MKLDNNMemoryInputNode*>(sibling);
        IE_ASSERT(inputNode != nullptr);
        node->setInputNode(inputNode);
    } else {
        holder[node->getId()] = node;
    }
    return &holder;
}

void MKLDNNMemoryNodeVirtualEdge::remove(MKLDNNMemoryNode * node, Holder* holder) {
    std::lock_guard<std::mutex> lock{MKLDNNMemoryNodeVirtualEdge::holderMutex};
    if (nullptr != holder) {
        InferenceEngine::details::erase_if(*holder, [&](const Holder::value_type & it){
            return it.second == node;
        });
    }
}

REG_MKLDNN_PRIM_FOR(MKLDNNMemoryInputNode, MemoryInput);
REG_MKLDNN_PRIM_FOR(MKLDNNMemoryOutputNode, MemoryOutput);
