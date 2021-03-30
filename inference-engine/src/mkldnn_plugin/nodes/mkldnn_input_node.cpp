// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_input_node.h"
#include "caseless.hpp"
#include "common/cpu_memcpy.h"
#include "../mkldnn_extension_utils.h"

#include <string>
#include <tuple>
#include <algorithm>
#include "caseless.hpp"
#include "common/cpu_memcpy.h"
#include "common/cpu_convert.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine::details;

MKLDNNInputNode::MKLDNNInputNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {
    constant = ConstantType::NoConst;
    if (layer && CaselessEq<std::string>()(layer->type, "const")) {
        constant = ConstantType::Const;
        if (layer->blobs.size() != 1
            || getType() != Input
            || !layer->blobs.begin()->second
            || layer->outData.empty())
            IE_THROW() << "Incorrect const input " << getName();
        cloneIfRequired(layer->blobs.begin()->second, layer->outData[0]->getTensorDesc());
    }
}

void MKLDNNInputNode::cloneIfRequired(const InferenceEngine::Blob::Ptr & blob, const InferenceEngine::TensorDesc & outTensorDesc) {
    ieConstBlob = blob;

    const InferenceEngine::TensorDesc& td = {blob->getTensorDesc().getPrecision(), outTensorDesc.getDims(), outTensorDesc.getLayout()};

    auto memDesc = MKLDNNMemoryDesc(td);

    auto cloneBlob = [&] () {
        MKLDNNMemory memory{ getEngine() };
        memory.Create(memDesc, blob->buffer());

        MKLDNNMemoryPtr ptr = MKLDNNMemoryPtr(new MKLDNNMemory(getEngine()));
        ptr->Create(memDesc);
        ptr->SetData(memory);

        return ptr;
    };

    auto isBlobAligned = [&] () {
        const void *ptr = blob->cbuffer().as<const void*>();
        size_t element_size = blob->element_size();
        return (reinterpret_cast<size_t>(ptr) % element_size) == 0;
    };

    // This code makes me cry, we should iterate over all elements of the blob.
    // The presence of subnormals is better to determined at IR read time.
    auto hasSubnormals = [&] () {
        if (blob->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32) {
            uint32_t const *u32data = blob->cbuffer().as<const uint32_t*>();
            for (size_t i = 0; i < blob->size(); ++i) {
                if (u32data[i] && (u32data[i] & (0xFF << 23)) == 0) {
                    return true;
                }
            }
        }
        return false;
    };

    if (weightCache) {
        char ptr[32];
        snprintf(ptr, sizeof ptr, "%p", blob->cbuffer().as<const void*>());
        const std::string key = getName()
                                    + "_" + std::to_string(blob->byteSize())
                                    + "_" + ptr;

        constBlob = *weightCache->findOrCreate(key, cloneBlob);
    } else if (isBlobAligned() && !hasSubnormals()) {
        constBlob = MKLDNNMemoryPtr(new MKLDNNMemory(getEngine()));
        constBlob->Create(memDesc, blob->buffer());
    } else {
        constBlob = cloneBlob();
    }
}

void MKLDNNInputNode::getSupportedDescriptors() {
    if (getType() == Input) {
        if (!getParentEdges().empty())
            IE_THROW() << "Incorrect number of input edges for layer " << getName();
        if (getChildEdges().empty())
            IE_THROW() << "Incorrect number of output edges for layer " << getName();
    } else if (getType() == Output) {
        if (getParentEdges().size() != 1)
            IE_THROW() << "Incorrect number of input edges for layer " << getName();
        if (!getChildEdges().empty())
            IE_THROW() << "Incorrect number of output edges for layer " << getName();
    }
}

void MKLDNNInputNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    if (getType() == Input || getType() == MemoryInput) {
        precision = getCnnLayer()->outData[0]->getPrecision();
        if (precision == InferenceEngine::Precision::U16 || isMeanImage) {
            precision = InferenceEngine::Precision::FP32;
        }
        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;

        auto mem_tdesc = MKLDNNMemoryDesc(getCnnLayer()->outData[0]->getTensorDesc());
        dataConfig.desc = mem_tdesc;
        config.outConfs.push_back(dataConfig);
    } else if (getType() == Output) {
        precision = getCnnLayer()->insData[0].lock()->getPrecision();
        if (precision == InferenceEngine::Precision::U16) precision = InferenceEngine::Precision::FP32;
        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;

        auto mem_tdesc = MKLDNNMemoryDesc(getCnnLayer()->insData[0].lock()->getTensorDesc());
        dataConfig.desc = mem_tdesc;
        config.inConfs.push_back(dataConfig);
    }
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void MKLDNNInputNode::createPrimitive() {
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto &dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
        if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
            IE_THROW() << "Destination memory didn't allocate for node " << getName()
                               << " to node " << getChildEdgeAt(i)->getChild()->getName() << ".";
    }
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto &srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
            IE_THROW() << "Destination memory didn't allocate for node " << getName()
                               << " from node " << getParentEdgeAt(i)->getParent()->getName() << ".";
    }

    const PrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";
}

bool MKLDNNInputNode::created() const {
    return getType() == Input || getType() == Output;
}

void MKLDNNInputNode::withMeanImage() {
    isMeanImage = true;
}

MKLDNNMemoryPtr MKLDNNInputNode::getConstBlob() const {
    return constBlob;
}

REG_MKLDNN_PRIM_FOR(MKLDNNInputNode, Input);
REG_MKLDNN_PRIM_FOR(MKLDNNInputNode, Output);
