// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "mkldnn_memory_node.hpp"
#include "common/simple_copy.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

std::mutex MKLDNNMemoryNodeVirtualEdge::holderMutex;

MKLDNNMemoryOutputNode::MKLDNNMemoryOutputNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) , MKLDNNMemoryNode(layer) {
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

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format::any);
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, memory::format::any);
}

const MKLDNNEdgePtr MKLDNNMemoryOutputNode::getChildEdgeAt(size_t idx) const {
    if (inputNode != nullptr) {
        return inputNode->getChildEdgeAt(idx);
    }
    return MKLDNNNode::getChildEdgeAt(idx);
}

void MKLDNNMemoryOutputNode::execute(mkldnn::stream strm)  {
    const uint8_t *src_ptr = getParentEdgeAt(0)->getBlob()->cbuffer().as<const uint8_t *>() +
            getParentEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
    uint8_t *dst_ptr = getChildEdgeAt(0)->getBlob()->cbuffer().as<uint8_t *>() +
            getChildEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

    size_t src_size = getParentEdgeAt(0)->getBlob()->byteSize();
    size_t dest_size = getChildEdgeAt(0)->getBlob()->byteSize();

    // TODO: this can be eliminated by completely removing MKLDNN memory output NODE, to fuse it with output of prev layer
    simple_copy(dst_ptr, dest_size, src_ptr, src_size);
}

#if defined (COMPILED_CPU_MKLDNN_INPUT_NODE)
MKLDNNMemoryInputNode::MKLDNNMemoryInputNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNInputNode(layer, eng, cache), MKLDNNMemoryNode(layer) {
    if (created()) {
        holder = MKLDNNMemoryNodeVirtualEdge::registerInput(this);
    }
}

MKLDNNMemoryInputNode::~MKLDNNMemoryInputNode() {
    MKLDNNMemoryNodeVirtualEdge::remove(this, holder);
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
#endif

MKLDNNMemoryNodeVirtualEdge::Holder* MKLDNNMemoryNodeVirtualEdge::registerOutput(MKLDNNMemoryOutputNode * node) {
    std::lock_guard<std::mutex> lock{MKLDNNMemoryNodeVirtualEdge::holderMutex};
    // in case of output layer
    auto& holder = MKLDNNMemoryNodeVirtualEdge::getExisted();
    auto sibling = MKLDNNMemoryNodeVirtualEdge::getByName(holder, node->getId());
    if (sibling != nullptr) {
#if defined (COMPILED_CPU_MKLDNN_INPUT_NODE)
        auto inputNode = dynamic_cast<MKLDNNMemoryInputNode*>(sibling);
        IE_ASSERT(inputNode != nullptr);
        node->setInputNode(inputNode);
#else
        THROW_IE_EXCEPTION << "CPU Plugin doesn't contain Input layer!";
#endif
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

#if defined (COMPILED_CPU_MKLDNN_INPUT_NODE)
REG_MKLDNN_PRIM_FOR(MKLDNNMemoryInputNode, MemoryInput);
#endif
REG_MKLDNN_PRIM_FOR(MKLDNNMemoryOutputNode, MemoryOutput);
