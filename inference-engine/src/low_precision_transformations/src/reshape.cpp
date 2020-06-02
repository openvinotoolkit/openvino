// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/reshape.hpp"

#include <algorithm>
#include <details/caseless.hpp>
#include <memory>
#include <string>
#include <vector>

#include "low_precision_transformations/common/ie_lpt_exception.hpp"
#include "low_precision_transformations/network_helper.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

size_t getChannelVolume(const SizeVector& dims) {
    size_t volume = 1ul;
    for (size_t i = 2; i < dims.size(); ++i) {
        volume = volume * dims[i];
    }

    return volume;
}

void ReshapeTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!canBeTransformed(context, layer)) {
        return;
    }

    if ((layer.insData.size() == 0) || layer.insData.size() > 2) {
        THROW_IE_EXCEPTION << "layer inputs '" << layer.insData.size() << "' is not correct";
    }

    if (!CaselessEq<std::string>()(layer.type, "Reshape")) {
        THROW_IE_EXCEPTION << "layer '" << layer.name << "' is not correct";
    }

    if (layer.insData.size() > 1) {
        transformOriginal(context, layer);
    } else {
        transformConstPropagated(context, layer);
    }
}

bool ReshapeTransformation::canTransformOriginal(const CNNLayer& layer) const {
    const CNNLayerPtr constLayer = CNNNetworkHelper::getParent(layer, 1);
    if (constLayer == nullptr) {
        THROW_IE_EXCEPTION << "Layer '" << layer.name << "' does not have parent at 1 position";
    }
    if (constLayer->type != "Const") {
        return false;
    }

    const Blob::Ptr paramsBlob = CNNNetworkHelper::getBlob(constLayer, "custom");
    const Precision precision = paramsBlob->getTensorDesc().getPrecision();
    if (!CNNNetworkHelper::isBlobPrecisionSupported(precision)) {
        THROW_IE_EXCEPTION << "layer " << constLayer->type << " '" << constLayer->name << "' unexpected precision " << precision;
    }

    if (paramsBlob->size() < 2) {
        return false;
    }

    const DataPtr inputData = layer.insData[0].lock();
    if (inputData == nullptr) {
        THROW_IE_EXCEPTION << "input data is absent";
    }

    const std::vector<size_t> inputDims = inputData->getTensorDesc().getDims();
    if (inputDims.size() < 2) {
        return false;
    }

    std::shared_ptr<float> paramsBufferData = CNNNetworkHelper::getFloatData(paramsBlob);
    float* params = paramsBufferData.get();
    if (((params[0] != -1) && (params[0] != 0) && (inputDims[0] != params[0])) ||
        ((params[1] != -1) && (params[1] != 0) && (inputDims[1] != params[1]))) {
        return false;
    }

    return true;
}

void ReshapeTransformation::transformOriginal(TransformationContext& context, CNNLayer& layer) const {
    if (!canTransformOriginal(layer)) {
        return;
    }

    const CNNLayerPtr constLayer = CNNNetworkHelper::getParent(layer, 1);
    const Blob::Ptr paramsBlob = CNNNetworkHelper::getBlob(constLayer, "custom");
    const signed int* paramsBuffer = paramsBlob->buffer().as<const signed int*>();
    if (paramsBuffer[1] == -1) {
        quantize(context, layer);
        return;
    }

    TransparentBaseTransformation::transform(context, layer);
}

bool ReshapeTransformation::canTransformConstPropagated(const CNNLayer& layer) const {
    if (layer.insData.size() != 1) {
        THROW_IE_EXCEPTION << "unexpected input count " << layer.insData.size();
    }
    const DataPtr input = layer.insData[0].lock();
    if (input == nullptr) {
        THROW_IE_EXCEPTION << "input is absent";
    }
    const std::vector<size_t> inputDims = input->getDims();
    if (inputDims.size() < 2) {
        return false;
    }

    if (layer.outData.size() != 1) {
        THROW_IE_EXCEPTION << "unexpected output count " << layer.outData.size();
    }
    const std::vector<size_t> outputDims = layer.outData[0]->getDims();
    if (outputDims.size() < 2) {
        return false;
    }

    const CNNLayerPtr dequantizationLayer = CNNNetworkHelper::getParent(layer, 0ul);
    if ((dequantizationLayer->outData[0]->getTensorDesc().getLayout() != Layout::NCHW) || (layer.outData[0]->getTensorDesc().getLayout() != Layout::NC)) {
        for (size_t i = 0; i < 2; ++i) {
            if (inputDims[i] != outputDims[i]) {
                return false;
            }
        }
    }

    return true;
}

void ReshapeTransformation::transformConstPropagated(TransformationContext& context, CNNLayer& layer) const {
    if (!canTransformConstPropagated(layer)) {
        return;
    }

    const CNNLayerPtr dequantizationLayer = CNNNetworkHelper::getParent(layer, 0ul);
    if ((dequantizationLayer->outData[0]->getTensorDesc().getLayout() == Layout::NCHW) && (layer.outData[0]->getTensorDesc().getLayout() == Layout::NC)) {
        quantize(context, layer);
        return;
    }

    TransparentBaseTransformation::transform(context, layer);
}

void ReshapeTransformation::quantize(TransformationContext& context, CNNLayer& layer) const {
    const CNNLayerPtr dequantizationLayer = CNNNetworkHelper::getParent(layer, 0ul);
    if ((dequantizationLayer == nullptr) || (dequantizationLayer->type != "ScaleShift")) {
        return;
    }

    const size_t inputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(*dequantizationLayer);
    const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(layer);
    const DataPtr insData = layer.insData[0].lock();
    if (insData == nullptr) {
        THROW_IE_LPT_EXCEPTION(layer) << "input data is absent";
    }
    const size_t channelVolume = getChannelVolume(insData->getTensorDesc().getDims());
    const DataPtr dequantizationDataPtr = dequantizationLayer->insData[0].lock();
    if (dequantizationDataPtr == nullptr) {
        THROW_IE_LPT_EXCEPTION(*dequantizationLayer) << "input data is absent";
    }
    if (insData->getTensorDesc().getDims()[0] != dequantizationDataPtr->getTensorDesc().getDims()[0] ||
        inputChannelsCount * channelVolume != outputChannelsCount)
        return;

    std::vector<float> originalDataDequantizationScales;
    std::vector<float> originalDataDequantizationShifts;
    fillFromDequantizationLayer(*dequantizationLayer, originalDataDequantizationScales, originalDataDequantizationShifts);

    std::vector<float> dequantizationScales(outputChannelsCount);
    std::vector<float> dequantizationShifts(outputChannelsCount);

    for (size_t inputChannel = 0ul; inputChannel < inputChannelsCount; inputChannel++) {
        for (size_t i = 0ul; i < channelVolume; i++) {
            dequantizationScales[inputChannel * channelVolume + i] = originalDataDequantizationScales[inputChannel];
            dequantizationShifts[inputChannel * channelVolume + i] = originalDataDequantizationShifts[inputChannel];
        }
    }

    if (updatePrecisions) {
        const Precision lowPrecision = getPrecisionBeforeParentDequantizationScaleShift(layer);
        CNNNetworkHelper::setOutDataPrecision(layer, lowPrecision);
    }

    CNNNetworkHelper::removeLayer(context.network, dequantizationLayer);
    context.removeLayer(*dequantizationLayer);

    addDequantizationLayer(context, layer, dequantizationScales, dequantizationShifts);
}

bool ReshapeTransformation::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
    return (layer.insData.size() > 1) ? canTransformOriginal(layer) : canTransformConstPropagated(layer);
}
