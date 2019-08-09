// Copyright (C) 2018-2019 Intel Corporation
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

MKLDNNQuantizeNode::MKLDNNQuantizeNode(InferenceEngine::CNNLayerPtr layer, const mkldnn::engine& eng, int socket) :
        MKLDNNNode(layer, eng, socket) {}

void MKLDNNQuantizeNode::initValues() {
    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        THROW_IE_EXCEPTION << "Quantize layer " << getName() << " supports only FP32 precision";

    auto* quantizeLayer = dynamic_cast<QuantizeLayer*>(getCnnLayer().get());
    if (quantizeLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert Quantize layer " << getName();

    levels = quantizeLayer->levels;
    if (levels <= 1)
        THROW_IE_EXCEPTION << "Quantize layer " << getName() << " supports only parameter levels > 1";

    size_t inputDataEdgeIdx = 0;
    size_t inputLowEdgeIdx = 0;
    size_t outputLowEdgeIdx = 0;
    size_t outputHighEdgeIdx = 0;
    auto parents = getParentEdges();
    for (size_t i = 0; i < parents.size(); i++) {
        auto p_edge = parents[i].lock();
        if (p_edge->getParent()->getType() == Input && p_edge->getParent()->getCnnLayer()->type == "Const") {
            inputLowEdgeIdx = i;
            outputLowEdgeIdx = i + 2;
            outputHighEdgeIdx = i + 3;
            inputDataEdgeIdx = i == 0 ? 4 : 0;
            break;
        }
    }

    for (size_t i = 0; i < parents.size(); i++) {
        auto p_edge = parents[i].lock();
        if (p_edge->getParent()->getType() == Input) {
            if (p_edge->getDims().ndims() != 1 && p_edge->getDims().ndims() != 4) {
                THROW_IE_EXCEPTION << "Quantize layer " << getName() << " supports only 1D or 4D inputs at edge " << i;
            }
        }
    }

    if (getParentEdges().size() != 5)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    if (getParentEdgeAt(inputDataEdgeIdx)->getDims().ndims() != 4) {
        THROW_IE_EXCEPTION << "Quantize layer " << getName() << " supports only 4D input at edge 0";
    }

    auto outputLowBlob = dynamic_cast<TBlob<float>*>(getParentEdgeAt(outputLowEdgeIdx)->getParent()->getCnnLayer()->blobs["custom"].get());
    auto outputLowData = outputLowBlob->buffer().as<float*>();
    int outputLowAxis = getParentEdgeAt(outputLowEdgeIdx)->getDims().ndims() == 1 ? 0 : 1;
    auto outputHighBlob = dynamic_cast<TBlob<float>*>(getParentEdgeAt(outputHighEdgeIdx)->getParent()->getCnnLayer()->blobs["custom"].get());
    auto outputHighData = outputHighBlob->buffer().as<float*>();
    int outputHighAxis = getParentEdgeAt(outputHighEdgeIdx)->getDims().ndims() == 1 ? 0 : 1;

    bool isBinarization = levels == 2;
    for (int i = 0; i < getParentEdgeAt(outputLowEdgeIdx)->getDims()[outputLowAxis]; i++) {
        if (outputLowData[i] != 1.f && outputLowData[i] != 0.f) {
            isBinarization = false;
            break;
        }
    }

    for (int i = 0; i < getParentEdgeAt(outputHighEdgeIdx)->getDims()[outputHighAxis]; i++) {
        if (outputHighData[i] != 1.f && outputHighData[i] != 0.f) {
            isBinarization = false;
            break;
        }
    }

    canStorePacked = isBinarization && getChildEdges().size() == 1 && getChildEdgeAt(0)->getChild()->getType() == BinaryConvolution;

    InferenceEngine::SizeVector dims;
    dims.push_back(getParentEdgeAt(inputDataEdgeIdx)->getDims()[1]);

    auto InputLowBlob = dynamic_cast<TBlob<float>*>(getParentEdgeAt(inputLowEdgeIdx)->getParent()->getCnnLayer()->blobs["custom"].get());

    auto inputLowData = InputLowBlob->buffer().as<float*>();
    int inputLowAxis = getParentEdgeAt(inputLowEdgeIdx)->getDims().ndims() == 1 ? 0 : 1;
    bool isInputLowBroadcasted = getParentEdgeAt(inputLowEdgeIdx)->getDims()[inputLowAxis] != dims[0];

    for (int i = 0; i < dims[0]; i++) {
        binarizationThresholds.push_back(inputLowData[isInputLowBroadcasted ? 0 : i]);
    }

    bool isOutputHighBroadcasted = getParentEdgeAt(outputHighEdgeIdx)->getDims()[outputHighAxis] != dims[0];
    for (int i = 0; i < dims[0]; i++) {
        uint32_t mask = outputHighData[isOutputHighBroadcasted ? 0 : i] == 1.f ? 0xffffffff : 0x00000000;

        binarizationOutputMask.push_back(mask);
    }

    initialized = true;
}

void MKLDNNQuantizeNode::getSupportedDescriptors() {
    if (isPackedStore()) {
        mkldnn::memory::data_type idt = MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision::FP32);
        mkldnn::memory::data_type ddt = MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision::BIN);
        mkldnn::memory::data_type wdt = MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision::FP32);
        mkldnn::memory::data_type omdt = MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision::FP32);

        MKLDNNMemoryDesc in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), idt, memory::nhwc);
        MKLDNNMemoryDesc out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), ddt, memory::nhwc);

        InferenceEngine::SizeVector weightDims;
        weightDims.push_back(getParentEdgeAt(0)->getDims()[1]);
        MKLDNNDims blocked_weightDims(weightDims);
        MKLDNNMemoryDesc wgh_candidate{blocked_weightDims, wdt, memory::x};
        MKLDNNMemoryDesc om_candidate{blocked_weightDims, omdt, memory::x};

        std::shared_ptr<mkldnn::binarization_forward::desc> bin_conv_desc;
        bin_conv_desc.reset(new binarization_forward::desc(prop_kind::forward_scoring, algorithm ::binarization_depthwise,
                                                           in_candidate, wgh_candidate, om_candidate, out_candidate));

        descs.emplace_back(bin_conv_desc);
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
        return {config, impl, fmt};
    };

    supportedPrimitiveDescriptors.push_back(same(memory::nhwc, ref_any));

    if (isPackedStore()) {
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
        binarizationDataMem->Create(binarizationDataDesc, getBinarizationTresholdsPtr());
        internalBlobMemory.push_back(binarizationDataMem);

        MKLDNNMemoryDesc binarizationMaskDataDesc = {{getParentEdgeAt(0)->getDims()[1]}, memory::f32, memory::x};
        auto binarizationMaskDataMem = std::make_shared<MKLDNNMemory>(getEngine());
        binarizationMaskDataMem->Create(binarizationMaskDataDesc, getBinarizationOutputMaskPtr());
        internalBlobMemory.push_back(binarizationMaskDataMem);

        prim.reset(new binarization_forward(prim_desc, getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                            internalBlobMemory[0]->GetPrimitive(),
                                            internalBlobMemory[1]->GetPrimitive(),
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
