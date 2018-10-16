// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_input_node.h"
#include "../mkldnn_extension_utils.h"
#include <string>

using namespace mkldnn;
using namespace MKLDNNPlugin;

MKLDNNInputNode::MKLDNNInputNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng) : MKLDNNNode(layer, eng) {}

void MKLDNNInputNode::getSupportedDescriptors() {
    if (getType() == Input) {
        if (!getParentEdges().empty())
            THROW_IE_EXCEPTION << "Incorrect number of input edges.";
        if (getChildEdges().empty())
            THROW_IE_EXCEPTION << "Incorrect number of output edges.";
    } else if (getType() == Output) {
        if (getParentEdges().size() != 1)
            THROW_IE_EXCEPTION << "Incorrect number of input edges.";
        if (!getChildEdges().empty())
            THROW_IE_EXCEPTION << "Incorrect number of output edges.";
    }
    constant = ConstantType::NoConst;
    auto layer = getCnnLayer();
    if (layer && CaselessEq<std::string>()(layer->type, "const")) {
        constant = ConstantType::Const;
        if (layer->blobs.size() != 1 || getType() != Input || !layer->blobs.begin()->second)
            THROW_IE_EXCEPTION << "Incorrect const input " << getName();
        constBlob = layer->blobs.begin()->second;
    }
}

void MKLDNNInputNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    if (getType() == Input || getType() == MemoryInput) {
        InferenceEngine::Precision precision = getCnnLayer()->outData[0]->getPrecision();
        if (precision == InferenceEngine::Precision::U16 || isMeanImage)
            precision = InferenceEngine::Precision::FP32;
        auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;

        memory::format layout = MKLDNNMemory::Convert(getCnnLayer()->outData[0]->getLayout());
        dataConfig.desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, layout);
        config.outConfs.push_back(dataConfig);
    } else if (getType() == Output) {
        InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
        if (precision == InferenceEngine::Precision::U16) precision = InferenceEngine::Precision::FP32;
        auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;

        memory::format layout = MKLDNNMemory::Convert(getCnnLayer()->insData[0].lock()->getLayout());
        dataConfig.desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, layout);
        config.inConfs.push_back(dataConfig);
    }
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void MKLDNNInputNode::createPrimitive() {
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto &dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
        if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Destination memory didn't allocate for node " << getName()
                               << " to node " << getChildEdgeAt(i)->getChild()->getName() << ".";
    }
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto &srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Destination memory didn't allocate for node " << getName()
                               << " from node " << getParentEdgeAt(i)->getParent()->getName() << ".";
    }

    const PrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set for node " << getName() << ".";
}

bool MKLDNNInputNode::created() const {
    return getType() == Input || getType() == Output;
}

void MKLDNNInputNode::execute(mkldnn::stream strm) {
    if (!constBlob)
        return;
    auto dstBlob = getChildEdgeAt(0)->getBlob();
    const float *srcData = constBlob->cbuffer().as<float *>();
    float *dstData = dstBlob->buffer();
    if (constBlob->size() != dstBlob->size()) {
        THROW_IE_EXCEPTION << "Incorrect blob sizes for node " << getName();
    }
    for (size_t i = 0; i < constBlob->size(); i++) {
        // srcData without offset() because constBlob should be planar
        dstData[dstBlob->getTensorDesc().offset(i)] = srcData[i];
    }
}
