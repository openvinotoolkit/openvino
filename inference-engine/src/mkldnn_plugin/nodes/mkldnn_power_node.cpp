// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_power_node.h"
#include "ie_layers.h"
#include <string>
#include <cmath>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <limits>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNPowerNode::MKLDNNPowerNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng)
        : MKLDNNNode(layer, eng), scale(1.0f), shift(1.0f), power(1.0f) {}

void MKLDNNPowerNode::getSupportedDescriptors() {
    auto * powerLayer = dynamic_cast<PowerLayer*>(getCnnLayer().get());

    if (powerLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert power layer.";
    scale = powerLayer->scale;
    power = powerLayer->power;
    shift = powerLayer->offset;

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";
}

void MKLDNNPowerNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;
    for (auto format : getAvailableFormatsForDims(getParentEdgeAt(0)->getDims())) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, format);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, format);
        if (format != memory::any) {
            config.inConfs[0].desc = InferenceEngine::TensorDesc(config.inConfs[0].desc.getPrecision(),
                                                                 config.inConfs[0].desc.getDims(), {
                                                                         config.inConfs[0].desc.getBlockingDesc().getBlockDims(),
                                                                         config.inConfs[0].desc.getBlockingDesc().getOrder(),
                                                                         std::numeric_limits<size_t>::max()
                                                                 });
            config.outConfs[0].desc = InferenceEngine::TensorDesc(config.outConfs[0].desc.getPrecision(),
                                                                  config.outConfs[0].desc.getDims(), {
                                                                          config.outConfs[0].desc.getBlockingDesc().getBlockDims(),
                                                                          config.outConfs[0].desc.getBlockingDesc().getOrder(),
                                                                          std::numeric_limits<size_t>::max()
                                                                  });
        }
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
    }
}

void MKLDNNPowerNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";
}

void MKLDNNPowerNode::execute(mkldnn::stream strm) {
    auto& srcMemory = getParentEdgeAt(0)->getMemory();
    auto& dstMemory = getChildEdgeAt(0)->getMemory();
    const size_t data_size = srcMemory.GetSize() / sizeof(float) / srcMemory.GetDims()[0] * batchToProcess();

    const auto *src_ptr = reinterpret_cast<const float*>(srcMemory.GetData()) +
            srcMemory.GetDescriptor().data.layout_desc.blocking.offset_padding;
    float *dst_ptr = reinterpret_cast<float*>(dstMemory.GetData()) +
            dstMemory.GetDescriptor().data.layout_desc.blocking.offset_padding;

    if (power == 1.0f) {
        #pragma omp parallel for
        for (int i = 0; i < data_size; i++)
            dst_ptr[i] = src_ptr[i] * scale + shift;
    } else {
        #pragma omp parallel for
        for (int i = 0; i < data_size; i++)
            dst_ptr[i] = pow(src_ptr[i] * scale + shift, power);
    }
}

bool MKLDNNPowerNode::created() const {
    return getType() == Power;
}
