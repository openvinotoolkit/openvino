// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_quantize_node.h"
#include "desc_iterator.hpp"
#include <legacy/ie_layers.h>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <algorithm>
#include <set>
#include <cmath>

// Quantization ranges validation is switched off by default in order to avoid regressions on user side
// #define VALIDATE_QUANTIZATION_RANGES

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

MKLDNNQuantizeNode::MKLDNNQuantizeNode(InferenceEngine::CNNLayerPtr layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(layer, eng, cache) {}

void MKLDNNQuantizeNode::init() {
    auto* quantizeLayer = dynamic_cast<QuantizeLayer*>(getCnnLayer().get());
    if (quantizeLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert Quantize layer " << getName();

    levels = quantizeLayer->levels;
    if (levels <= 1)
        THROW_IE_EXCEPTION << "Quantize layer " << getName() << " supports only parameter levels > 1";

    if (getParentEdges().size() != 5)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    for (size_t i = 0; i < getParentEdges().size(); i++) {
        if (getParentEdgesAtPort(i).size() != 1)
            THROW_IE_EXCEPTION << "Quantize layer " << getName() << " has unsupported number of parent edges at port " << i;
    }

    if (getParentEdgesAtPort(0)[0]->getDims().ndims() < 1ul || getParentEdgesAtPort(0)[0]->getDims().ndims() > 5ul) {
        THROW_IE_EXCEPTION << "Unsupported number of dimensions for input at edge 0 in Quantize layer " << getName();
    }

    auto initAxisIdx = [&](size_t edgeIdx) {
        auto edge = getParentEdgesAtPort(edgeIdx)[0];

        size_t axisIdx = 0;
        int numberOfNonUnit = 0;
        if (edge->getDims().ndims() > 0) {
            if (edge->getDims()[0] > 1) {
                numberOfNonUnit++;
            }
        }

        for (int i = 1; i < edge->getDims().ndims(); i++) {
            if (edge->getDims()[i] > 1) {
                axisIdx = i;
                numberOfNonUnit++;
            }
        }
        if (numberOfNonUnit > 1) {
            THROW_IE_EXCEPTION << "Quantize layer " << getName() << " supports only per-tensor and per-channel quantizations";
        }

        return axisIdx;
    };

    axis = getParentEdgesAtPort(0)[0]->getDims().ndims() == 1 ? 0 : 1;

    std::set<size_t> quantizationParamsAxisesIdxs;
    std::set<size_t> quantizationParamsAxisesSizes;

    auto inputLowAxis = initAxisIdx(1);
    isInputLowBroadcasted = getParentEdgesAtPort(1)[0]->getDims()[inputLowAxis] == 1;
    if (!isInputLowBroadcasted) {
        quantizationParamsAxisesIdxs.insert(inputLowAxis);
        quantizationParamsAxisesSizes.insert(getParentEdgesAtPort(1)[0]->getDims()[inputLowAxis]);
    }

    auto inputHighAxis = initAxisIdx(2);
    isInputHighBroadcasted = getParentEdgesAtPort(2)[0]->getDims()[inputHighAxis] == 1;
    if (!isInputHighBroadcasted) {
        quantizationParamsAxisesIdxs.insert(inputHighAxis);
        quantizationParamsAxisesSizes.insert(getParentEdgesAtPort(2)[0]->getDims()[inputHighAxis]);
    }

    auto outputLowAxis = initAxisIdx(3);
    isOutputLowBroadcasted = getParentEdgesAtPort(3)[0]->getDims()[outputLowAxis] == 1;
    if (!isOutputLowBroadcasted) {
        quantizationParamsAxisesIdxs.insert(outputLowAxis);
        quantizationParamsAxisesSizes.insert(getParentEdgesAtPort(3)[0]->getDims()[outputLowAxis]);
    }

    auto outputHighAxis = initAxisIdx(4);
    isOutputHighBroadcasted = getParentEdgesAtPort(4)[0]->getDims()[outputHighAxis] == 1;
    if (!isOutputHighBroadcasted) {
        quantizationParamsAxisesIdxs.insert(outputHighAxis);
        quantizationParamsAxisesSizes.insert(getParentEdgesAtPort(4)[0]->getDims()[outputHighAxis]);
    }

    if (quantizationParamsAxisesIdxs.size() > 1 || quantizationParamsAxisesSizes.size() > 1)
        THROW_IE_EXCEPTION << "Unsupported input sizes for Quantize layer with name " << getName();

    if (quantizationParamsAxisesIdxs.size() == 1) {
        axis = *quantizationParamsAxisesIdxs.begin();
    }

    auto inputLowAxisSize = getParentEdgesAtPort(1)[0]->getDims()[inputLowAxis];
    auto inputHighAxisSize = getParentEdgesAtPort(2)[0]->getDims()[inputHighAxis];
    auto outputLowAxisSize = getParentEdgesAtPort(3)[0]->getDims()[outputLowAxis];
    auto outputHighAxisSize = getParentEdgesAtPort(4)[0]->getDims()[outputHighAxis];

    size_t axisRealSize = static_cast<size_t>(getParentEdgesAtPort(0)[0]->getDims()[axis]);
    size_t axisPaddedSize = static_cast<size_t>(rnd_up(getParentEdgesAtPort(0)[0]->getDims()[axis], 16));

    if (quantizationParamsAxisesSizes.size() == 1) {
        if (*quantizationParamsAxisesSizes.begin() != axisRealSize)
            THROW_IE_EXCEPTION << "Unsupported input sizes for Quantize layer with name " << getName();
    }

    for (size_t i = 1; i < getParentEdges().size(); i++) {
        if (!getParentEdgesAtPort(i)[0]->getParent()->isConstant())
            THROW_IE_EXCEPTION << "Quantize layer with name " << getName() << " has non const input on " << i << " port";
        auto prec = getCnnLayer()->insData[i].lock()->getPrecision();
        if (prec != Precision::FP32)
            THROW_IE_EXCEPTION << "Quantize layer with name " << getName() << " has unsupported precision " << prec << " on " << i << " port";
    }

    auto inputLowBlob = dynamic_cast<TBlob<float>*>(getParentEdgesAtPort(1)[0]->getParent()->getCnnLayer()->blobs["custom"].get());
    auto inputLowData = inputLowBlob->buffer().as<float*>();

    auto inputHighBlob = dynamic_cast<TBlob<float>*>(getParentEdgesAtPort(2)[0]->getParent()->getCnnLayer()->blobs["custom"].get());
    auto inputHighData = inputHighBlob->buffer().as<float*>();

    auto outputLowBlob = dynamic_cast<TBlob<float>*>(getParentEdgesAtPort(3)[0]->getParent()->getCnnLayer()->blobs["custom"].get());
    auto outputLowData = outputLowBlob->buffer().as<float*>();

    auto outputHighBlob = dynamic_cast<TBlob<float>*>(getParentEdgesAtPort(4)[0]->getParent()->getCnnLayer()->blobs["custom"].get());
    auto outputHighData = outputHighBlob->buffer().as<float*>();

    bool binarization = levels == 2;

    if (binarization) {
        for (int i = 0; i < outputLowAxisSize; i++) {
            if (outputLowData[i] != 1.f && outputLowData[i] != 0.f) {
                binarization = false;
                break;
            }
        }

        for (int i = 0; i < outputHighAxisSize; i++) {
            if (outputHighData[i] != 1.f && outputHighData[i] != 0.f) {
                binarization = false;
                break;
            }
        }

        for (ptrdiff_t i = 0; i < std::max(inputLowAxisSize, inputHighAxisSize); i++) {
            if (inputLowData[isInputLowBroadcasted ? 0 : i] != inputHighData[isInputHighBroadcasted ? 0 : i]) {
                binarization = false;
                break;
            }
        }
    }

    if (binarization) {
        quantizeAlgorithm = algorithm::binarization_depthwise;

        binarizationThresholds.resize(axisPaddedSize);
        binarizationOutputMask.resize(axisPaddedSize);

        for (int i = 0; i < axisRealSize; i++) {
            binarizationThresholds[i] = inputLowData[isInputLowBroadcasted ? 0 : i];
            binarizationOutputMask[i] = outputHighData[isOutputHighBroadcasted ? 0 : i] == 1.f ? 0xffffffff : 0x00000000;
        }
    } else {
        auto allElementsAreEqual = [&](const float* data, size_t size) {
            if (size == 0)
                return true;

            auto first = data[0];
            for (int i = 1; i < size; i++) {
                if (data[i] != first)
                    return false;
            }

            return true;
        };

        if (allElementsAreEqual(inputLowData, inputLowAxisSize)) {
            inputLowAxisSize = 1;
            isInputLowBroadcasted = true;
        }

        if (allElementsAreEqual(inputHighData, inputHighAxisSize)) {
            inputHighAxisSize = 1;
            isInputHighBroadcasted = true;
        }

        if (allElementsAreEqual(outputLowData, outputLowAxisSize)) {
            outputLowAxisSize = 1;
            isOutputLowBroadcasted = true;
        }

        if (allElementsAreEqual(outputHighData, outputHighAxisSize)) {
            outputHighAxisSize = 1;
            isOutputHighBroadcasted = true;
        }

        cropLow.resize(inputLowAxisSize);
        cropHigh.resize(inputHighAxisSize);
        inputScale.resize(std::max(inputLowAxisSize, inputHighAxisSize));
        inputShift.resize(std::max(inputLowAxisSize, inputHighAxisSize));
        outputScale.resize(std::max(outputLowAxisSize, outputHighAxisSize));
        outputShift.resize(outputLowAxisSize);

        bool quantizationOnly = true;

        for (int i = 0; i < cropLow.size(); i++) {
            float il = inputLowData[isInputLowBroadcasted ? 0 : i];

            cropLow[i] = il;
        }

        for (int i = 0; i < cropHigh.size(); i++) {
            float ih = inputHighData[isInputHighBroadcasted ? 0 : i];

            cropHigh[i] = ih;
        }

        for (int i = 0; i < inputScale.size(); i++) {
            float il = inputLowData[isInputLowBroadcasted ? 0 : i];
            float ih = inputHighData[isInputHighBroadcasted ? 0 : i];

#if defined(VALIDATE_QUANTIZATION_RANGES)
            if ((il == ih && levels != 2) || std::isnan(il) || std::isnan(ih) || std::isinf(il) || std::isinf(ih)) {
                THROW_IE_EXCEPTION << "Quantize layer with name '" << getName() << "' has invalid input quantize ranges: "
                                   << "inputLow = " << il << ", inputHigh = " << ih;
            }
#endif

            inputScale[i] = (levels - 1) / (ih - il);
            inputShift[i] = -il * (levels - 1) / (ih - il);
        }

        for (int i = 0; i < outputScale.size(); i++) {
            float ol = outputLowData[isOutputLowBroadcasted ? 0 : i];
            float oh = outputHighData[isOutputHighBroadcasted ? 0 : i];

#if defined(VALIDATE_QUANTIZATION_RANGES)
            if (std::isnan(ol) || std::isnan(oh) || std::isinf(ol) || std::isinf(oh)) {
                THROW_IE_EXCEPTION << "Quantize layer with name '" << getName() << "' has wrong output quantize ranges: "
                                   << "outputLow = " << ol << ", outputHigh = " << oh;
            }
#endif

            outputScale[i] = (oh - ol) / (levels - 1);

            if (outputScale[i] != 1.f)
                quantizationOnly = false;
        }

        for (int i = 0; i < outputShift.size(); i++) {
            float ol = outputLowData[isOutputLowBroadcasted ? 0 : i];

            outputShift[i] = ol;

            if (outputShift[i] != 0.f)
                quantizationOnly = false;
        }

        quantizeAlgorithm = quantizationOnly ? algorithm::quantization_quantize : algorithm::quantization_quantize_dequantize;
    }

    if (binarization) {
        inputPrecision = Precision::FP32;
        outputPrecision = Precision::BIN;
    } else {
        inputPrecision = getCnnLayer()->insData[0].lock()->getPrecision();
        outputPrecision = getCnnLayer()->outData[0]->getPrecision();

        if (inputPrecision != Precision::FP32 && inputPrecision != Precision::U8 && inputPrecision != Precision::I8)
            inputPrecision = Precision::FP32;

        if (outputPrecision != Precision::FP32 && outputPrecision != Precision::U8 && outputPrecision != Precision::I8)
            outputPrecision = Precision::FP32;
    }
}

std::vector<mkldnn::memory::format> MKLDNNQuantizeNode::getDataFormats() const {
    // Special case for first FQ in the network
    if (getParentEdgesAtPort(0)[0]->getDims()[getAxis()] == 3) {
        return { MKLDNNMemory::GetPlainFormat(getParentEdgesAtPort(0)[0]->getDims()) };
    } else {
        if (isBinarization()) {
            if (getParentEdgesAtPort(0)[0]->getDims().ndims() == 4)
                return {memory::nhwc };
            else
                return { MKLDNNMemory::GetPlainFormat(getParentEdgesAtPort(0)[0]->getDims()) };
        } else {
            switch (getParentEdgesAtPort(0)[0]->getDims().ndims()) {
                case 2:
                    return {memory::nc};
                case 4:
                    return {memory::nChw8c, memory::nChw16c, memory::nhwc, memory::nchw};
                case 5:
                    return {memory::nCdhw8c, memory::nCdhw16c, memory::ndhwc, memory::ncdhw};
                default:
                    return {MKLDNNMemory::GetPlainFormat(getParentEdgesAtPort(0)[0]->getDims())};
            }
        }
    }
}

void MKLDNNQuantizeNode::getSupportedDescriptors() {
    mkldnn::memory::data_type idt = MKLDNNExtensionUtils::IEPrecisionToDataType(getInputPrecision());
    mkldnn::memory::data_type wdt = MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision::FP32);
    mkldnn::memory::data_type ddt = MKLDNNExtensionUtils::IEPrecisionToDataType(getOutputPrecision());

    for (auto& format : getDataFormats()) {
        MKLDNNMemoryDesc in_candidate = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), idt, format);
        MKLDNNMemoryDesc out_candidate = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), ddt, format);

        InferenceEngine::SizeVector weightDims;
        weightDims.push_back(getParentEdgeAt(0)->getDims()[getAxis()]);
        MKLDNNDims blocked_weightDims(weightDims);
        MKLDNNMemoryDesc wgh_candidate{blocked_weightDims, wdt, memory::x};

        if (isBinarization()) {
            std::shared_ptr<mkldnn::quantization_forward::desc> bin_conv_desc;
            bin_conv_desc.reset(new quantization_forward::desc(prop_kind::forward_scoring, quantizeAlgorithm, getAxis(),
                                                               in_candidate, wgh_candidate, wgh_candidate,
                                                               out_candidate));

            descs.emplace_back(bin_conv_desc);
        } else if (levels != 2) {
            std::shared_ptr<mkldnn::quantization_forward::desc> quantization_desc;
            quantization_desc.reset(
                    new quantization_forward::desc(prop_kind::forward_scoring, quantizeAlgorithm, getAxis(),
                                                   in_candidate, wgh_candidate, wgh_candidate, wgh_candidate,
                                                   wgh_candidate, wgh_candidate, wgh_candidate, out_candidate));

            descs.emplace_back(quantization_desc);
        }
    }
}

void MKLDNNQuantizeNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getInputPrecision());
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getOutputPrecision());

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
                dataConfig.desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDataType, MKLDNNMemory::GetPlainFormat(getParentEdgeAt(i)->getDims()));
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

    if (!descs.empty()) {
        for (int i = 0; i < descs.size(); i++) {
            primitive_desc_iterator itpd = descs[i].createPrimitiveDescriptorIterator(getEngine());
            while (itpd.is_not_end()) {
                impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str());

                supportedPrimitiveDescriptors.push_back(same(getDataFormats()[i], impl_type));

                itpd++;
            }
        }
    } else {
        inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision::FP32);
        outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision::FP32);

        // Ref implementation. Not from MKLDNN.
        supportedPrimitiveDescriptors.push_back(same(MKLDNNMemory::GetPlainFormat(getParentEdgeAt(0)->getDims()), ref_any));
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

    size_t axisSize = getParentEdgeAt(0)->getDims()[getAxis()];
    size_t axisPaddedSize = rnd_up(axisSize, 16);
    MKLDNNMemoryDesc weightsDataDesc = {{(uint32_t)axisPaddedSize}, memory::f32, memory::x};

    if (isBinarization()) {
        auto prim_desc = createPrimitiveDescriptor<quantization_forward::primitive_desc, quantization_forward::desc>();

        auto binarizationThresholdsDataMem = std::make_shared<MKLDNNMemory>(getEngine());
        binarizationThresholdsDataMem->Create(weightsDataDesc, getBinarizationTresholdsPtr());
        internalBlobMemory.push_back(binarizationThresholdsDataMem);

        auto binarizationMaskDataMem = std::make_shared<MKLDNNMemory>(getEngine());
        binarizationMaskDataMem->Create(weightsDataDesc, getBinarizationOutputMaskPtr());
        internalBlobMemory.push_back(binarizationMaskDataMem);

        prim.reset(new quantization_forward(prim_desc, getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                            internalBlobMemory[0]->GetPrimitive(),
                                            internalBlobMemory[1]->GetPrimitive(),
                                            getChildEdgeAt(0)->getMemory().GetPrimitive()));
    } else if (levels != 2) {
        auto prim_desc = createPrimitiveDescriptor<quantization_forward::primitive_desc, quantization_forward::desc>();

        auto pushInternalBlob = [&](std::vector<float>& data) {
            if (data.size() == 1)
                data.resize(axisPaddedSize, data[0]);
            else
                data.resize(axisPaddedSize);
            auto memory = std::make_shared<MKLDNNMemory>(getEngine());
            memory->Create(weightsDataDesc, &data[0]);
            internalBlobMemory.push_back(memory);
        };

        pushInternalBlob(cropLow);
        pushInternalBlob(cropHigh);
        pushInternalBlob(inputScale);
        pushInternalBlob(inputShift);
        pushInternalBlob(outputScale);
        pushInternalBlob(outputShift);

        prim.reset(new quantization_forward(prim_desc, getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                            internalBlobMemory[0]->GetPrimitive(),
                                            internalBlobMemory[1]->GetPrimitive(),
                                            internalBlobMemory[2]->GetPrimitive(),
                                            internalBlobMemory[3]->GetPrimitive(),
                                            internalBlobMemory[4]->GetPrimitive(),
                                            internalBlobMemory[5]->GetPrimitive(),
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

        auto srcDims = srcMemory->GetDims();
        srcDims[0] = batchToProcess();
        if (axis >= srcDims.size())
            THROW_IE_EXCEPTION << "Axis " << axis << " exceeds source tensor dimensions number";

        size_t outerSize = 1;
        for (size_t i = 0; i < axis; i++)
            outerSize *= srcDims[i];

        size_t axisSize = srcDims[axis];

        size_t innerSize = 1;
        for (size_t i = axis + 1; i < srcDims.size(); i++)
            innerSize *= srcDims[i];

        size_t outerOffset = axisSize * innerSize;
        size_t axisOffset = innerSize;

        for (size_t ou = 0; ou < outerSize; ou++) {
            for (size_t ax = 0; ax < axisSize; ax++) {
                float inputLow = inputLowData[isInputLowBroadcasted ? 0 : ax];
                float inputHigh = inputHighData[isInputHighBroadcasted ? 0 : ax];
                float outputLow = outputLowData[isOutputLowBroadcasted ? 0 : ax];
                float outputHigh = outputHighData[isOutputHighBroadcasted ? 0 : ax];

                for (size_t is = 0; is < innerSize; is++) {
                    size_t idx = ou * outerOffset + ax * axisOffset + is;

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

void MKLDNNQuantizeNode::appendPostOps(mkldnn::post_ops& ops) {
    if (!isPostOpDataInitialized) {
        isPostOpDataInitialized = true;
        cropLowData.set(cropLow.size(), 1 << 1, &cropLow[0]);
        cropHighData.set(cropHigh.size(), 1 << 1, &cropHigh[0]);
        inputScaleData.set(inputScale.size(), 1 << 1, &inputScale[0]);
        inputShiftData.set(inputShift.size(), 1 << 1, &inputShift[0]);
        outputScaleData.set(outputScale.size(), 1 << 1, &outputScale[0]);
        outputShiftData.set(outputShift.size(), 1 << 1, &outputShift[0]);
    }

    ops.append_quantization(quantizeAlgorithm, &cropLowData, &cropHighData, &inputScaleData, &inputShiftData, &outputScaleData, &outputShiftData);
}

bool MKLDNNQuantizeNode::created() const {
    return getType() == Quantize;
}
REG_MKLDNN_PRIM_FOR(MKLDNNQuantizeNode, Quantize);
