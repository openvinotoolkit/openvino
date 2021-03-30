// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "mkldnn_memory_node.hpp"
#include "common/cpu_memcpy.h"
#include "mkldnn_graph.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNMemoryOutputNode::MKLDNNMemoryOutputNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) , MKLDNNMemoryNode(layer) {
}

void MKLDNNMemoryOutputNode::init() {
    if (created()) {
        // in case of output layer
        IE_ASSERT(graph != nullptr);
        auto itNode = graph->virtualEdge.find(getId());
        if (itNode != graph->virtualEdge.end()) {
            auto inputNode = dynamic_cast<MKLDNNMemoryInputNode*>(itNode->second);
            IE_ASSERT(inputNode != nullptr);
            setInputNode(inputNode);
        } else {
            graph->virtualEdge.emplace(getId(), this);
        }
    }
}

MKLDNNMemoryOutputNode::~MKLDNNMemoryOutputNode() {
    if (graph != nullptr) {
        InferenceEngine::details::erase_if(graph->virtualEdge, [&](const std::pair<std::string, MKLDNNMemoryNode*>& it){
            return it.second == this;
        });
    }
}

void MKLDNNMemoryOutputNode::getSupportedDescriptors() {}

void MKLDNNMemoryOutputNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, MKLDNNMemory::GetPlainFormat(getParentEdgeAt(0)->getDims()));
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, memory::format_tag::any);
}

void MKLDNNMemoryOutputNode::execute(mkldnn::stream strm)  {
    auto& srcMemory = getParentEdgeAt(0)->getMemory();

    auto inputMemoryNode = dynamic_cast<MKLDNNMemoryInputNode*>(inputNode);
    IE_ASSERT(inputMemoryNode != nullptr);
    inputMemoryNode->storeState(srcMemory);
}

MKLDNNMemoryInputNode::MKLDNNMemoryInputNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNInputNode(layer, eng, cache), MKLDNNMemoryNode(layer), dataStore(new MKLDNNMemory{eng}) {
}

void MKLDNNMemoryInputNode::init() {
    if (created()) {
        IE_ASSERT(graph != nullptr);
        // in case of output already registered
        auto itNode = graph->virtualEdge.find(getId());
        if (itNode != graph->virtualEdge.end()) {
            auto outputNode = dynamic_cast<MKLDNNMemoryOutputNode*>(itNode->second);
            IE_ASSERT(outputNode != nullptr);
            outputNode->setInputNode(this);
        } else {
            graph->virtualEdge.emplace(getId(), this);
        }
    }
}

MKLDNNMemoryInputNode::~MKLDNNMemoryInputNode() {
    if (graph != nullptr) {
        InferenceEngine::details::erase_if(graph->virtualEdge, [&](const std::pair<std::string, MKLDNNMemoryNode*>& it){
            return it.second == this;
        });
    }
}

void MKLDNNMemoryInputNode::createPrimitive() {
    MKLDNNInputNode::createPrimitive();

    auto mem_desc = getChildEdgeAt(0)->getMemoryPtr()->GetDescriptor();
    dataStore->Create(mem_desc);

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
static void simple_copy(MKLDNNMemory& dst, const MKLDNNMemory& src) {
    auto srcPtr = static_cast<uint8_t*>(src.GetPtr());
    auto dstPtr = static_cast<uint8_t*>(dst.GetPtr());
    auto srcSizeInByte = src.GetSize();
    auto dstSizeInByte = dst.GetSize();

    IE_ASSERT(srcSizeInByte == dstSizeInByte) << "Memory objects are not compatible. Has different sizes.";

    cpu_memcpy(dstPtr, srcPtr, srcSizeInByte);
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
    auto dst_mem = getChildEdgeAt(0)->getMemory();
    // TODO: Should be simple call of:
    //           dst_mem.SetData(dataStore, false);
    //       But because of performance reason we use simple manual copy
    simple_copy(dst_mem, *dataStore);
}

REG_MKLDNN_PRIM_FOR(MKLDNNMemoryInputNode, MemoryInput);
REG_MKLDNN_PRIM_FOR(MKLDNNMemoryOutputNode, MemoryOutput);
