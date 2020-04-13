// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize.hpp"

#include <algorithm>
#include <blob_factory.hpp>
#include <cmath>
#include <details/caseless.hpp>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <details/ie_cnn_network_tools.h>
#include <ie_common.h>
#include <precision_utils.h>
#include "cnn_network_impl.hpp"
#include "ie_util_internal.hpp"
#include "low_precision_transformations/common/ie_lpt_exception.hpp"
#include "low_precision_transformations/network_helper.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

void FakeQuantizeTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!CaselessEq<std::string>()(layer.type, "FakeQuantize")) {
        THROW_IE_EXCEPTION << "Layer '" << layer.name << "' has invalid type. FakeQuantize is expected.";
    }

    if (layer.insData.size() != 5lu) {
        THROW_IE_EXCEPTION << "Layer '" << layer.insData.size() << "' has invalid inputs number. 5 is expected.";
    }

    // CNNNetworkHelper::invertFakeQuantize(layer);

    // FakeQuantize on weights are used without dequantization ScaleShifts
    const bool onWeights = CNNNetworkHelper::onWeights(layer);
    if (onWeights) {
        return;
    }

    if (!QuantizationDetails::outputLayoutIsSupported(layer)) {
        return;
    }

    CNNLayerPtr fakeQuantizeLayer = std::make_shared<CNNLayer>(layer);
    CNNLayerPtr scaleShift = CNNNetworkHelper::getParent(layer, 0);
    auto scaleShiftChildren = CNNNetworkHelper::getChildren(*scaleShift);
    if ((scaleShift != nullptr) && (scaleShift->type == "ScaleShift") && scaleShiftChildren.size() == 1) {
        fuseScaleShift(context, fakeQuantizeLayer, scaleShift);
    }

    if (context.quantizedFakeQuantizeNames.find(layer.name) != context.quantizedFakeQuantizeNames.end()) {
        return;
    }

    if (!QuantizationDetails::isSupportedLevel(layer.GetParamAsUInt("levels"))) return;

    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(layer);
    const DataPrecision dataPrecision = getDataPrecision(layer, quantizationDetails, onWeights, supportAsymmetricQuantization);
    if (dataPrecision.precision == Precision::UNSPECIFIED) {
        return;
    }

    std::vector<float> dequantizationScales;
    std::vector<float> dequantizationShifts;
    fillFromQuantizationDetails(
        quantizationDetails,
        dataPrecision,
        dequantizationScales,
        dequantizationShifts);

#ifdef LPT_PRINT_DEQUANTIZATION_INFO
    printDequantizationValues(dequantizationScales, dequantizationShifts);
#endif

    CNNNetworkHelper::updateBlobs(layer, 3, dataPrecision.min);
    CNNNetworkHelper::updateBlobs(layer, 4, dataPrecision.max);

    if (updatePrecisions) {
        CNNNetworkHelper::setOutDataPrecision(layer, dataPrecision.precision);
    }

    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(layer);
    if (children.size() == 0) {
        const std::string originalName = layer.name;
        CNNNetworkHelper::renameLayer(context.network, layer.name, layer.name + LayerTransformation::lastLayerPrefix);

        CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
            context, std::make_shared<CNNLayer>(layer), nullptr,
            DequantizationDetails(dequantizationScales, dequantizationShifts, dequantizationShifts.size()),
            originalName);
        context.dequantizationLayersNames.insert(dequantizationLayer->name);
    } else {
        for (const CNNLayerPtr& child : children) {
            CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context, std::make_shared<CNNLayer>(layer), child,
                DequantizationDetails(dequantizationScales, dequantizationShifts, dequantizationShifts.size()));
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
        }
    }

    context.quantizedFakeQuantizeNames.insert(layer.name);
}

bool FakeQuantizeTransformation::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
    return false;
}

void FakeQuantizeTransformation::fuseScaleShift(TransformationContext& context, CNNLayerPtr fakeQuantizeLayer,
                                                CNNLayerPtr scaleShift) const {
    // TODO: add check if previous blobs precision is enough to store current values
    const Blob::Ptr scalesBlob = CNNNetworkHelper::getBlob(scaleShift, "weights");
    std::shared_ptr<float> scalesBufferPtr = CNNNetworkHelper::getFloatData(scalesBlob);

    const Blob::Ptr shiftsBlob = CNNNetworkHelper::getBlob(scaleShift, "biases");
    std::shared_ptr<float> shiftsBufferPtr = CNNNetworkHelper::getFloatData(shiftsBlob);

    if (scalesBlob->size() != shiftsBlob->size())
        THROW_IE_EXCEPTION << "Scales and shifts values count are different for " << scaleShift->name;

    const float* shiftsBuffer = shiftsBufferPtr.get();
    const float* scalesBuffer = scalesBufferPtr.get();
    // Don't fuse when there is a negative scale, because it leads to invalid results of FQ
    for (size_t i = 0lu; i < scalesBlob->size(); ++i) {
        if (scalesBuffer[i] <= 0.0f) return;
    }

    CNNLayerPtr inputLow = CNNNetworkHelper::getParent(*fakeQuantizeLayer, 1);
    CNNLayerPtr inputHigh = CNNNetworkHelper::getParent(*fakeQuantizeLayer, 2);

    Layout layout;
    size_t channelIndex;
    const DataPtr insData = scaleShift->insData[0].lock();
    if (insData == nullptr) {
        THROW_IE_LPT_EXCEPTION(*scaleShift) << "input data is absent";
    }
    const size_t inputDims = insData->getDims().size();
    switch (inputDims) {
        case 5: {
            layout = Layout::NCDHW;
            channelIndex = 1ul;
            break;
        }
        case 4: {
            layout = Layout::NCHW;
            channelIndex = 1ul;
            break;
        }
        case 3: {
            layout = Layout::BLOCKED;
            channelIndex = 1ul;
            break;
        }
        case 2: {
            layout = Layout::NC;
            channelIndex = 1ul;
            break;
        }
        case 1: {
            layout = Layout::C;
            channelIndex = 0ul;
            break;
        }
        default: THROW_IE_EXCEPTION << "FakeQuantizeTransform: unexpected dimensions count " << inputDims << " in ScaleShift optimization";
    }
    std::vector<size_t> dims(inputDims, 1lu);
    dims[channelIndex] = scalesBlob->size();

    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(*fakeQuantizeLayer);

    Blob::Ptr targetInputLowBufferPtr = reshapeWeightsIntervalConst(*inputLow, dims, layout);
    auto targetInputLowBuffer = CNNNetworkHelper::getFloatData(targetInputLowBufferPtr);
    Blob::Ptr targetInputHighBufferPtr = reshapeWeightsIntervalConst(*inputHigh, dims, layout);
    auto targetInputHighBuffer = CNNNetworkHelper::getFloatData(targetInputHighBufferPtr);

    for (size_t i = 0lu; i < scalesBlob->size(); ++i) {
        auto q_lo = quantizationDetails.getInputLowValue(i);
        auto q_hi = quantizationDetails.getInputHighValue(i);
        auto sc = scalesBlob->size() == 1 ? scalesBuffer[0] : scalesBuffer[i];
        auto sh = shiftsBlob->size() == 1 ? shiftsBuffer[0] : shiftsBuffer[i];
        targetInputLowBuffer.get()[i] = (q_lo - sh) / sc;
        targetInputHighBuffer.get()[i] = (q_hi - sh) / sc;
    }

    CNNNetworkHelper::fillBlobByFP32(targetInputLowBufferPtr, targetInputLowBuffer.get());
    CNNNetworkHelper::fillBlobByFP32(targetInputHighBufferPtr, targetInputHighBuffer.get());

    reshapeFakeQuantize(*fakeQuantizeLayer, dims, layout);

    CNNNetworkHelper::removeLayer(context.network, scaleShift);
    context.removeLayer(*scaleShift);
}

Blob::Ptr FakeQuantizeTransformation::reshapeWeightsIntervalConst(CNNLayer& constLayer, const std::vector<size_t>& dims,
                                                                  const Layout layout) {
    if (constLayer.blobs.size() != 1lu) {
        THROW_IE_EXCEPTION << "Unexpected blobs count " << constLayer.blobs.size() << " for layer " << constLayer.name;
    }
    if (constLayer.outData.size() != 1lu)
        THROW_IE_EXCEPTION << "Unexpected outputs for layer " << constLayer.name;

    auto it = constLayer.blobs.find("custom");
    if (it == constLayer.blobs.end()) THROW_IE_EXCEPTION << "blob 'custom' was not found for layer " << constLayer.name;

    const Precision& srcPrecision = it->second->getTensorDesc().getPrecision();

    Blob::Ptr targetBlob = CNNNetworkHelper::makeNewBlobPtr({srcPrecision, dims, layout});
    targetBlob->allocate();
    constLayer.blobs["custom"] = targetBlob;

    constLayer.outData[0]->reshape(dims, layout);

    return targetBlob;
}

void FakeQuantizeTransformation::reshapeFakeQuantize(
        CNNLayer& fakeQuantizeLayer,
        const std::vector<size_t>& dims,
        const Layout layout) {
    DataPtr inputLowData = fakeQuantizeLayer.insData[1].lock();
    if (inputLowData == nullptr) {
        THROW_IE_EXCEPTION << "input low interval data is absent";
    }
    inputLowData->reshape(dims, layout);

    DataPtr inputHighData = fakeQuantizeLayer.insData[2].lock();
    if (inputHighData == nullptr) {
        THROW_IE_EXCEPTION << "input hight interval data is absent";
    }
    inputHighData->reshape(dims, layout);
}
