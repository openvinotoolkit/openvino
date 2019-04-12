// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_quantize_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <ie_memcpy.h>
#include "details/caseless.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

MKLDNNQuantizeNode::MKLDNNQuantizeNode(InferenceEngine::CNNLayerPtr layer, const mkldnn::engine& eng) : MKLDNNNode(layer, eng) {}

void MKLDNNQuantizeNode::getSupportedDescriptors() {
    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        THROW_IE_EXCEPTION << "Quantize layer " << getName() << " supports only FP32 precision";

    auto* quantizeLayer = dynamic_cast<QuantizeLayer*>(getCnnLayer().get());
    if (quantizeLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert Quantize layer " << getName();

    levels = quantizeLayer->levels;
    if (levels <= 1)
        THROW_IE_EXCEPTION << "Quantize layer " << getName() << "supports only parameter levels > 1";

    if (getParentEdges().size() != 5)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    if (getParentEdgeAt(0)->getDims().ndims() != 4) {
        THROW_IE_EXCEPTION << "Quantize layer " << getName() << "supports only 4D input at edge 0";
    }

    for (int i = 1; i < 5; i++) {
        if (getParentEdgeAt(i)->getDims().ndims() != 1 && getParentEdgeAt(i)->getDims().ndims() != 4) {
            THROW_IE_EXCEPTION << "Quantize layer " << getName() << "supports only 1D or 4D inputs at edge " << i;
        }
    }

    canStorePacked = getChildEdges().size() == 1 && getChildEdgeAt(0)->getChild()->getType() == BinaryConvolution;

    if (canStorePacked) {
        mkldnn::memory::data_type idt = MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision::FP32);
        mkldnn::memory::data_type ddt = MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision::BIN);
        mkldnn::memory::data_type wdt = MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision::FP32);

        MKLDNNMemoryDesc in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), idt, memory::nhwc);
        MKLDNNMemoryDesc out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), ddt, memory::nhwc);

        InferenceEngine::SizeVector weightDims;
        weightDims.push_back(getParentEdgeAt(0)->getDims()[1]);
        MKLDNNDims blocked_weightDims(weightDims);
        MKLDNNMemoryDesc wgh_candidate{blocked_weightDims, wdt, memory::x};


        std::shared_ptr<mkldnn::binarization_forward::desc> bin_conv_desc;
        bin_conv_desc.reset(new binarization_forward::desc(prop_kind::forward_scoring, algorithm::binarization_depthwise,
                                                           in_candidate, wgh_candidate, out_candidate));

        descs.emplace_back(bin_conv_desc);

        InferenceEngine::SizeVector dims;
        dims.push_back(getParentEdgeAt(0)->getDims()[1]);

        auto InputLowBlob = dynamic_cast<TBlob<float>*>(getParentEdgeAt(1)->getParent()->getCnnLayer()->blobs["custom"].get());

        auto inputLowData = InputLowBlob->buffer().as<float*>();
        int inputLowAxis = getParentEdgeAt(1)->getDims().ndims() == 1 ? 0 : 1;
        bool isInputLowBroadcasted = getParentEdgeAt(1)->getDims()[inputLowAxis] != dims[0];

        for (int i = 0; i < dims[0]; i++) {
            binarizationThresholds.push_back(inputLowData[isInputLowBroadcasted ? 0 : i]);
        }
    }
}

void MKLDNNQuantizeNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision::FP32);
    auto outputDataType = canStorePacked ? MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision::BIN)
                                         : MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision::FP32);



    auto same = [&] (memory::format fmt, impl_desc_type impl) -> PrimitiveDescInfo {
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = true;
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = -1;
            dataConfig.constant = false;

            if (i == 0) {
                dataConfig.desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDataType, fmt);
            } else {
                dataConfig.desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDataType,
                        getParentEdgeAt(i)->getDims().ndims() == 1 ? memory::x : memory::nchw);
            }
            config.inConfs.push_back(dataConfig);
        }

        InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = -1;
            dataConfig.constant = false;
            dataConfig.desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, fmt);
            config.outConfs.push_back(dataConfig);
        return {config, impl};
    };

    supportedPrimitiveDescriptors.push_back(same(memory::nhwc, ref_any));

    if (canStorePacked) {
        primitive_desc_iterator itpd = descs[0].createPrimitiveDescriptorIterator(getEngine());
        do {
            impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str());
            supportedPrimitiveDescriptors.push_back(same(memory::nhwc, impl_type));
        } while (itpd.next());
    }
}

void MKLDNNQuantizeNode::createPrimitive() {
    if (prim)
        return;

    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory isn't allocated.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory isn't allocated.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor isn't set.";

    if (canStorePacked) {
        auto prim_desc = createPrimitiveDescriptor<binarization_forward::primitive_desc, binarization_forward::desc>();

        MKLDNNMemoryDesc binarizationDataDesc = {{getParentEdgeAt(0)->getDims()[1]}, memory::f32, memory::x};
        auto binarizationDataMem = std::make_shared<MKLDNNMemory>(getEngine());
        binarizationDataMem->Create(binarizationDataDesc, &binarizationThresholds[0]);
        internalBlobMemory.push_back(binarizationDataMem);

        prim.reset(new binarization_forward(prim_desc, getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                            internalBlobMemory[0]->GetPrimitive(),
                                            getChildEdgeAt(0)->getMemory().GetPrimitive()));
    }
}

void MKLDNNQuantizeNode::execute(mkldnn::stream strm) {
    if (prim) {
        MKLDNNNode::execute(strm);
    } else {
        auto &srcMemory = getParentEdgeAt(0)->getMemoryPtr();
        auto &inputLowMemory = getParentEdgeAt(1)->getMemoryPtr();
        auto &inputHighMemory = getParentEdgeAt(2)->getMemoryPtr();
        auto &outputLowMemory = getParentEdgeAt(3)->getMemoryPtr();
        auto &outputHighMemory = getParentEdgeAt(4)->getMemoryPtr();
        auto &dstMemory = getChildEdgeAt(0)->getMemoryPtr();

        auto srcData = reinterpret_cast<const float *>(srcMemory->GetData());
        auto inputLowData = reinterpret_cast<const float *>(inputLowMemory->GetData());
        auto inputHighData = reinterpret_cast<const float *>(inputHighMemory->GetData());
        auto outputLowData = reinterpret_cast<const float *>(outputLowMemory->GetData());
        auto outputHighData = reinterpret_cast<const float *>(outputHighMemory->GetData());
        auto dstData = reinterpret_cast<float *>(dstMemory->GetData());

        srcData += srcMemory->GetDescriptor().data.layout_desc.blocking.offset_padding;
        inputLowData += inputLowMemory->GetDescriptor().data.layout_desc.blocking.offset_padding;
        inputHighData += inputHighMemory->GetDescriptor().data.layout_desc.blocking.offset_padding;
        outputLowData += outputLowMemory->GetDescriptor().data.layout_desc.blocking.offset_padding;
        outputHighData += outputHighMemory->GetDescriptor().data.layout_desc.blocking.offset_padding;
        dstData += dstMemory->GetDescriptor().data.layout_desc.blocking.offset_padding;

        size_t N = static_cast<size_t>(batchToProcess());
        size_t C = static_cast<size_t>(srcMemory->GetDims()[1]);
        size_t H = static_cast<size_t>(srcMemory->GetDims()[2]);
        size_t W = static_cast<size_t>(srcMemory->GetDims()[3]);

        int inputLowAxis = inputLowMemory->GetDims().size() == 1 ? 0 : 1;
        bool isInputLowBroadcasted = inputLowMemory->GetDims()[inputLowAxis] != C;

        int inputHighAxis = inputHighMemory->GetDims().size() == 1 ? 0 : 1;
        bool isInputHighBroadcasted = inputHighMemory->GetDims()[inputHighAxis] != C;

        int outputLowAxis = outputLowMemory->GetDims().size() == 1 ? 0 : 1;
        bool isOutputLowBroadcasted = outputLowMemory->GetDims()[outputLowAxis] != C;

        int outputHighAxis = outputHighMemory->GetDims().size() == 1 ? 0 : 1;
        bool isOutputHighBroadcasted = outputHighMemory->GetDims()[outputHighAxis] != C;

        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    for (int c = 0; c < C; c++) {
                        size_t idx = n * H * W * C + h * W * C + w * C + c;

                        float inputLow = inputLowData[isInputLowBroadcasted ? 0 : c];
                        float inputHigh = inputHighData[isInputHighBroadcasted ? 0 : c];
                        float outputLow = outputLowData[isOutputLowBroadcasted ? 0 : c];
                        float outputHigh = outputHighData[isOutputHighBroadcasted ? 0 : c];

                        if (srcData[idx] <= inputLow)
                            dstData[idx] = outputLow;
                        else if (srcData[idx] > inputHigh)
                            dstData[idx] = outputHigh;
                        else
                            dstData[idx] = roundf((srcData[idx] - inputLow) / (inputHigh - inputLow) * (levels - 1)) /
                                           (levels - 1) * (outputHigh - outputLow) + outputLow;
                    }
                }
            }
        }
    }
}

bool MKLDNNQuantizeNode::created() const {
    return getType() == Quantize;
}
