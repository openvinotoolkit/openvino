// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_reshape_node.h"
#include <ie_layers.h>
#include <string>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNReshapeNode::MKLDNNReshapeNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(layer, eng, cache) {}

void MKLDNNReshapeNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 1 && getParentEdges().size() != 2)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();
}

void MKLDNNReshapeNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    // Current reshape implementation is simple memory reinterpret,
    // same precision on input and output is required
    if (inputDataType != outputDataType)
        inputDataType = outputDataType;

    auto& outDims = getChildEdgeAt(0)->getDims();
    memory::format outFormat = MKLDNNMemory::GetPlainFormat(outDims);
    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(getParentEdges().size());
    for (size_t i = 0; i <getParentEdges().size(); i++) {
        config.inConfs[i].inPlace = -1;
        config.inConfs[i].constant = false;
        config.inConfs[i].desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDataType,
                                                  MKLDNNMemory::GetPlainFormat(getParentEdgeAt(i)->getDims()));
    }
    config.outConfs.resize(1);
    config.outConfs[0].inPlace = 0;
    config.outConfs[0].constant = false;
    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, outFormat);
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, outFormat);
}

void MKLDNNReshapeNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";
}

bool MKLDNNReshapeNode::created() const {
    return getType() == Reshape || getType() == Flatten;
}

#if GraphGen(Gen_Reshape)
REG_MKLDNN_PRIM_FOR(MKLDNNReshapeNode, Reshape);
#endif
