// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_depthwise_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNDepthwiseNode::MKLDNNDepthwiseNode(InferenceEngine::CNNLayerPtr layer, const mkldnn::engine& eng) : MKLDNNNode(layer, eng) {
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(0).desc());
    });
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        if (!isWithBiases())
            return MKLDNNMemoryDesc();
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(1).desc());
    });
}

void MKLDNNDepthwiseNode::getSupportedDescriptors() {
    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto parentOutDims = getParentEdgeAt(0)->getDims();

    SizeVector weightDims = { (long unsigned int)parentOutDims[1] };
    MKLDNNDims blocked_weightDims(weightDims);

    internalBlobs.push_back(createInternalBlob(weightDims, true));
    if (isWithBiases())
        internalBlobs.push_back(createInternalBlob(weightDims, false));

    for (auto format : getAvailableFormatsForDims(parentOutDims)) {
        MKLDNNMemoryDesc in_candidate{parentOutDims, inputDataType, format};
        createDescriptor({in_candidate}, {});
    }
}

void MKLDNNDepthwiseNode::createPrimitive() {
    if (prim)
        return;

    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";

    auto prim_desc = createPrimitiveDescriptor<depthwise_forward::primitive_desc, depthwise_forward::desc>();

    if (isBroadcast()) {
        float broadcastValue = static_cast<float*>(internalBlobMemory[0]->GetData())[0];
        for (int i = 1; i < internalBlobMemory[0]->GetPrimitiveDescriptor().desc().data.dims[0]; i++) {
            static_cast<float*>(internalBlobMemory[0]->GetData())[i] = broadcastValue;
        }

        if (isWithBiases()) {
            broadcastValue = static_cast<float*>(internalBlobMemory[1]->GetData())[0];
            for (int i = 1; i < internalBlobMemory[1]->GetPrimitiveDescriptor().desc().data.dims[0]; i++) {
                static_cast<float*>(internalBlobMemory[1]->GetData())[i] = broadcastValue;
            }
        }
    }

    if (isWithBiases()) {
        prim.reset(new depthwise_forward(prim_desc, getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                         internalBlobMemory[0]->GetPrimitive(),
                                         internalBlobMemory[1]->GetPrimitive(),
                                         getChildEdgeAt(0)->getMemory().GetPrimitive()));
    } else {
        prim.reset(new depthwise_forward(prim_desc, getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                         internalBlobMemory[0]->GetPrimitive(),
                                         getChildEdgeAt(0)->getMemory().GetPrimitive()));
    }
}

bool MKLDNNDepthwiseNode::created() const {
    return getType() == Depthwise;
}

void MKLDNNDepthwiseNode::initValues() {
    GenericLayer* depthwiseLayer = getCnnLayer().get();
    if (depthwiseLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get CNNLayer.";

    CaselessEq<std::string> comparator;
    if (comparator(depthwiseLayer->type, "ScaleShift")) {
        auto *scshLayer = dynamic_cast<ScaleShiftLayer*>(getCnnLayer().get());
        if (scshLayer->_weights == nullptr)
            THROW_IE_EXCEPTION << "ScaleShift without weights is not supported";

        algorithm = depthwise_scale_shift;
        withBiases = scshLayer->_biases != nullptr;
        broadcast = static_cast<bool>(scshLayer->_broadcast);
    } else if (comparator(depthwiseLayer->type, "PReLU")) {
        auto *preluLayer = dynamic_cast<PReLULayer*>(getCnnLayer().get());
        if (preluLayer->_weights == nullptr)
            THROW_IE_EXCEPTION << "PReLU without weights is not supported";

        algorithm = depthwise_prelu;
        withBiases = false;
        broadcast = preluLayer->_channel_shared;
    } else {
        THROW_IE_EXCEPTION << "Unsupported depthwise operation";
    }

    initialized = true;
}

void MKLDNNDepthwiseNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                           const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    MKLDNNMemoryDesc in_candidate(inputDesc[0]);
    MKLDNNMemoryDesc out_candidate(inputDesc[0]);
    MKLDNNDims weightDims({in_candidate.getDims()[1]});

    if (in_candidate.getFormat() == memory::nChw16c) {
        weightDims[0] = rnd_up(weightDims[0], 16);
    } else if (in_candidate.getFormat() == memory::nChw8c) {
        weightDims[0] = rnd_up(weightDims[0], 8);
    }

    MKLDNNMemoryDesc wgh_candidate{weightDims, in_candidate.getDataType(), memory::x};

    if (isWithBiases()) {
        MKLDNNMemoryDesc bias_candidate{weightDims, in_candidate.getDataType(), memory::x};
        MKLDNNDescriptor desc(std::shared_ptr<depthwise_forward::desc>(
                new depthwise_forward::desc(prop_kind::forward_scoring, getAlgorithm(), in_candidate, out_candidate, wgh_candidate, bias_candidate)));
        descs.push_back(desc);
    } else {
        MKLDNNDescriptor desc(std::shared_ptr<depthwise_forward::desc>(
                new depthwise_forward::desc(prop_kind::forward_scoring, getAlgorithm(), in_candidate, out_candidate, wgh_candidate)));
        descs.push_back(desc);
    }
}

void MKLDNNDepthwiseNode::initOptimalPrimitiveDescriptor() {
    auto config = getSelectedPrimitiveDescriptor()->getConfig();
    if (isInitConfig(config))
        return;

    if (config.inConfs.size() != 1 || config.outConfs.size() != 1 || (!isUninitTensorDesc(config.inConfs[0].desc) &&
            !isUninitTensorDesc(config.outConfs[0].desc) && config.inConfs[0].desc != config.outConfs[0].desc))
        THROW_IE_EXCEPTION << "Layer " << getName() << " has incorrect selected config!";

    if (!isUninitTensorDesc(config.inConfs[0].desc)) {
        config.outConfs[0].desc = config.inConfs[0].desc;
    } else if (!isUninitTensorDesc(config.outConfs[0].desc)) {
        config.inConfs[0].desc = config.outConfs[0].desc;
    } else {
        config.outConfs[0].desc = config.inConfs[0].desc = getConfiguredInputDesc(config, 0);
    }

    initDescriptor(config);
}
