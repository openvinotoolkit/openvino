// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "mkldnn_memory_node.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNMemoryOutputNode::MKLDNNMemoryOutputNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, int socket)
        : MKLDNNNode(layer, eng, socket) , MKLDNNMemoryNode(layer) {
    if (created()) {
        MKLDNNMemoryNodeVirtualEdge::registerOutput(this);
    }
}

MKLDNNMemoryOutputNode::~MKLDNNMemoryOutputNode() {
    MKLDNNMemoryNodeVirtualEdge::remove(this);
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
    auto& srcMemory = getParentEdgeAt(0)->getMemory();

    const float *src_ptr = reinterpret_cast<const float*>(srcMemory.GetData()) +
            srcMemory.GetDescriptor().data.layout_desc.blocking.offset_padding;
    float *dst_ptr = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemory().GetData()) +
            getChildEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

    // TODO: this can be eliminated by completely removing MKLDNN memory output NODE, to fuse it with output of prev layer
    memcpy(dst_ptr, src_ptr, srcMemory.GetSize());
}

#if defined (COMPILED_CPU_MKLDNN_INPUT_NODE)
MKLDNNMemoryInputNode::MKLDNNMemoryInputNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, int socket)
        : MKLDNNInputNode(layer, eng, socket), MKLDNNMemoryNode(layer) {
    if (created()) {
        MKLDNNMemoryNodeVirtualEdge::registerInput(this);
    }
}

MKLDNNMemoryInputNode::~MKLDNNMemoryInputNode() {
    MKLDNNMemoryNodeVirtualEdge::remove(this);
}

void MKLDNNMemoryNodeVirtualEdge::registerInput(MKLDNNMemoryInputNode * node) {
    // in case of output already registered
    auto sibling = MKLDNNMemoryNodeVirtualEdge::getByName(node->getId());
    if (sibling != nullptr) {
        auto outputNode = dynamic_cast<MKLDNNMemoryOutputNode*>(sibling);
        IE_ASSERT(outputNode != nullptr);
        outputNode->setInputNode(node);
    } else {
        getExisted()[node->getId()] = node;
    }
}
#endif

void MKLDNNMemoryNodeVirtualEdge::registerOutput(MKLDNNMemoryOutputNode * node) {
    // in case of output layer
    auto sibling = MKLDNNMemoryNodeVirtualEdge::getByName(node->getId());
    if (sibling != nullptr) {
#if defined (COMPILED_CPU_MKLDNN_INPUT_NODE)
        auto inputNode = dynamic_cast<MKLDNNMemoryInputNode*>(sibling);
        IE_ASSERT(inputNode != nullptr);
        node->setInputNode(inputNode);
#else
        THROW_IE_EXCEPTION << "CPU Plugin doesn't contain Input layer!";
#endif
    } else {
        getExisted()[node->getId()] = node;
    }
}
#if defined (COMPILED_CPU_MKLDNN_INPUT_NODE)
REG_MKLDNN_PRIM_FOR(MKLDNNMemoryInputNode, MemoryInput);
#endif
REG_MKLDNN_PRIM_FOR(MKLDNNMemoryOutputNode, MemoryOutput);
