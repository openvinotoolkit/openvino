// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/eltwise.hpp"
#include "low_precision_transformations/network_helper.hpp"

#include <details/ie_cnn_network_tools.h>
#include <ie_common.h>

#include <algorithm>
#include <details/caseless.hpp>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ie_util_internal.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

bool EltwiseTransformation::canBeTransformed(const TransformationContext& context, const CNNLayer& layer) const {
    if (!LayerTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    if (!CaselessEq<std::string>()(layer.type, "Eltwise")) {
        THROW_IE_EXCEPTION << "layer type '" << layer.name << "' is not correct";
    }

    const TensorDesc& tensorDesc0 = layer.insData[0].lock()->getTensorDesc();
    for (size_t i = 1ul; i < layer.insData.size(); ++i) {
        const auto& data = layer.insData[i];
        if (!isSupported(tensorDesc0, data.lock()->getTensorDesc())) {
            return false;
        }
    }

    return true;
}

void EltwiseTransformation::transform(TransformationContext& context, CNNLayer& eltwise) const {
    if (!canBeTransformed(context, eltwise)) {
        return;
    }

    if ((!eltwise.CheckParamPresence("operation")) || (eltwise.GetParamAsString("operation") != "sum")) {
        return;
    }

    if (!CaselessEq<std::string>()(eltwise.type, "Eltwise")) {
        THROW_IE_EXCEPTION << "layer type '" << eltwise.name << "' is not correct";
    }

    if ((eltwise.insData.size() < 2) || (!eltwise.CheckParamPresence("operation")) ||
        (eltwise.GetParamAsString("operation") != "sum")) {
        return;
    }

    std::vector<CNNLayerPtr> quantizeLayers;
    const size_t numberInputLayers = eltwise.insData.size();
    const size_t outputChannelCount = CNNNetworkHelper::getOutputChannelsCount(eltwise);

    std::vector<std::string> childNameOurAfterQuantizeLayers;
    childNameOurAfterQuantizeLayers.resize(numberInputLayers);

    size_t quantizationLevels = 0lu;
    std::vector<QuantizationDetails> quantizationLayersDetails;
    for (int index = 0; index < numberInputLayers; index++) {
        DataPtr quantizeOnData = eltwise.insData[index].lock();
        if (quantizeOnData == nullptr) {
            THROW_IE_EXCEPTION << "input is absent";
        }

        auto layer = quantizeOnData->getCreatorLayer().lock();
        if ((layer->type != "FakeQuantize") && (layer->type != "Quantize")) {
            do {
                if (layer->type == "Pooling") {
                    childNameOurAfterQuantizeLayers[index] = layer->name;
                    layer = CNNNetworkHelper::getParent(*layer, 0);
                } else {
                    return;
                }
            } while ((layer->type != "FakeQuantize") && (layer->type != "Quantize"));
        } else {
            childNameOurAfterQuantizeLayers[index] = eltwise.name;
        }

        const QuantizationDetails& quantizationDetails = QuantizationDetails::getDetails(*layer);
        if (!QuantizationDetails::isSupportedLevel(quantizationDetails.levels)) continue;
        if (quantizationLevels == 0) {
            quantizationLevels = quantizationDetails.levels;
        } else if (quantizationLevels != quantizationDetails.levels) {
            THROW_IE_EXCEPTION << "different quantization levels " << quantizationLevels << " are not supported";
        }

        quantizeLayers.push_back(layer);
        quantizationLayersDetails.push_back(quantizationDetails);
    }

    if (quantizeLayers.empty()) {
        return;
    }

    const DataPrecision dataPrecision = getDataPrecision(*quantizeLayers[0], QuantizationDetails::getDetails(*quantizeLayers[0]), false, false);
    if (dataPrecision.precision == Precision::UNSPECIFIED) {
        return;
    }

    std::vector<float> dequantizationScales(outputChannelCount);
    std::vector<float> dequantizationShifts(outputChannelCount);

    std::vector<std::vector<float>> dequantizationShiftsLayers;
    dequantizationShiftsLayers.resize(numberInputLayers);

    // TODO: refactor: use cycle anyway
    // TODO: hardcode detected: zero element

    if ((quantizationLayersDetails[0].outputHighValues.size() == 1)) {
        std::vector<float> outputInterval;
        std::vector<float> lowNewOutput;
        float sumLowOldOutput = 0.f;
        for (int index = 0; index < numberInputLayers; index++) {
            const float outputLowValue = quantizationLayersDetails[index].outputLowValues[0];
            const float outputHighValue = quantizationLayersDetails[index].outputHighValues[0];
            outputInterval.push_back((outputHighValue - outputLowValue));
            sumLowOldOutput += outputLowValue;
        }

        if (quantizedTensorAlignmentOnActivations == QuantizedTensorAlignment::UpdateLevel) {
            const size_t minLevels = getMinQuantizationLevels(dataPrecision, outputInterval);
            if (minLevels < this->minQuantizationLevels) {
                return;
            }
        }

        const float maxOutputInterval = *std::max_element(outputInterval.begin(), outputInterval.end());
        if (maxOutputInterval == 0.f)
            THROW_IE_EXCEPTION << "Invalid output interval: " << maxOutputInterval;

        if (quantizedTensorAlignmentOnActivations == QuantizedTensorAlignment::UpdateLevel) {
            for (int index = 0; index < numberInputLayers; index++) {
                const int outputIntervalMax =
                    static_cast<int>(roundf(dataPrecision.max * outputInterval[index] / maxOutputInterval));
                if (outputIntervalMax <= INTERVALS_THRESHOLD) return;
            }
        }

        for (int index = 0; index < numberInputLayers; index++) {
            if (quantizeLayers[index] == nullptr)
                continue;
            CNNLayer& fakeQuantizeLayer = *quantizeLayers[index];

            // TODO: copy/paste, refactor: extract to MultiBranchTransformation::updateQuantizationRange
            switch (quantizedTensorAlignmentOnActivations) {
            case QuantizedTensorAlignment::None: {
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3,
                                              dataPrecision.min * outputInterval[index] / maxOutputInterval);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4,
                                              dataPrecision.max * outputInterval[index] / maxOutputInterval);
                break;
            }
            case QuantizedTensorAlignment::UpdateIntervals: {
                const float k = maxOutputInterval / outputInterval[index];
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 2,
                                              quantizationLayersDetails[index].inputHighValues[0] * k);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3,
                                              dataPrecision.min * outputInterval[index] / maxOutputInterval);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4, dataPrecision.max);
                break;
            }
            case QuantizedTensorAlignment::UpdateLevel: {
                const float outputIntervalMin = roundf(dataPrecision.min * outputInterval[index] / maxOutputInterval);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3, outputIntervalMin);

                const float outputIntervalMax = roundf(dataPrecision.max * outputInterval[index] / maxOutputInterval);
                CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4, outputIntervalMax);

                const size_t levels = static_cast<size_t>(fabs(outputIntervalMax - outputIntervalMin)) + 1ul;
                fakeQuantizeLayer.params["levels"] = std::to_string(levels);
                QuantizeLayer* layer = dynamic_cast<QuantizeLayer*>(&fakeQuantizeLayer);
                if (layer == nullptr) {
                    THROW_IE_EXCEPTION << "incorrect type for layer " << fakeQuantizeLayer.name;
                }
                layer->levels = levels;
                break;
            }
            default: {
                THROW_IE_EXCEPTION << "unexpected value " << quantizedTensorAlignmentOnActivations;
            }
            }

            if (updatePrecisions) {
                CNNNetworkHelper::setOutDataPrecision(fakeQuantizeLayer, dataPrecision.precision);
            }

            lowNewOutput.push_back(dataPrecision.min * outputInterval[index] / maxOutputInterval);

            context.quantizedFakeQuantizeNames.insert(quantizeLayers[index]->name);
        }

        float generalScaleDequantize = maxOutputInterval / (dataPrecision.max - dataPrecision.min);
        const float quantizationShift =
            (sumLowOldOutput - generalScaleDequantize * accumulate(lowNewOutput.begin(), lowNewOutput.end(), 0.0)) * -1;

        for (size_t channel = 0; channel < outputChannelCount; ++channel) {
            dequantizationScales[channel] = generalScaleDequantize;
            dequantizationShifts[channel] = -1 * quantizationShift;
            for (int index = 0; index < numberInputLayers; index++) {
                dequantizationShiftsLayers[index].push_back((quantizationLayersDetails[index].outputLowValues[0] -
                                                             generalScaleDequantize * lowNewOutput[index]));
            }
        }
    } else {
        for (size_t channel = 0; channel < outputChannelCount; ++channel) {
            std::vector<float> outputInterval;
            std::vector<float> lowNewOutput;
            float sumLowOldOutput = 0;
            for (int index = 0; index < numberInputLayers; index++) {
                const float outputLowValue = quantizationLayersDetails[index].getOutputLowValue(channel);
                const float outputHighValue = quantizationLayersDetails[index].getOutputHighValue(channel);
                outputInterval.push_back((outputHighValue - outputLowValue));
                sumLowOldOutput += outputLowValue;
            }

            if (quantizedTensorAlignmentOnActivations == QuantizedTensorAlignment::UpdateLevel) {
                const size_t minLevels = getMinQuantizationLevels(dataPrecision, outputInterval);
                if (minLevels < this->minQuantizationLevels) {
                    return;
                }
            }

            const float maxOutputInterval = *max_element(outputInterval.begin(), outputInterval.end());
            if (maxOutputInterval == 0.f)
                THROW_IE_EXCEPTION << "Invalid output interval: " << maxOutputInterval;

            if (quantizedTensorAlignmentOnActivations == QuantizedTensorAlignment::UpdateLevel) {
                for (int index = 0; index < numberInputLayers; index++) {
                    const int outputIntervalMax =
                        static_cast<int>(roundf(dataPrecision.max * outputInterval[index] / maxOutputInterval));
                    if (outputIntervalMax <= INTERVALS_THRESHOLD) return;
                }
            }

            for (int index = 0; index < numberInputLayers; index++) {
                CNNLayer& fakeQuantizeLayer = *quantizeLayers[index];

                // TODO: copy/paste, refactor: extract to MultiBranchTransformation::updateQuantizationRange
                switch (quantizedTensorAlignmentOnActivations) {
                case QuantizedTensorAlignment::None: {
                    CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3,
                                                  dataPrecision.min * outputInterval[index] / maxOutputInterval);
                    CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4,
                                                  dataPrecision.max * outputInterval[index] / maxOutputInterval);
                    break;
                }
                case QuantizedTensorAlignment::UpdateIntervals: {
                    const float k = maxOutputInterval / outputInterval[index];
                    CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 2,
                                                  quantizationLayersDetails[index].inputHighValues[0] * k);
                    CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3,
                                                  dataPrecision.min * outputInterval[index] / maxOutputInterval);
                    CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4, dataPrecision.max);
                    break;
                }
                case QuantizedTensorAlignment::UpdateLevel: {
                    const float outputIntervalMin = roundf(dataPrecision.min * outputInterval[index] / maxOutputInterval);
                    CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 3, outputIntervalMin);

                    const float outputIntervalMax = roundf(dataPrecision.max * outputInterval[index] / maxOutputInterval);
                    CNNNetworkHelper::updateBlobs(fakeQuantizeLayer, 4, outputIntervalMax);

                    const size_t levels = static_cast<size_t>(fabs(outputIntervalMax - outputIntervalMin)) + 1ul;
                    fakeQuantizeLayer.params["levels"] = std::to_string(levels);
                    QuantizeLayer* layer = dynamic_cast<QuantizeLayer*>(&fakeQuantizeLayer);
                    if (layer == nullptr) {
                        THROW_IE_EXCEPTION << "incorrect type for layer " << fakeQuantizeLayer.name;
                    }
                    layer->levels = levels;
                    break;
                }
                default: {
                    THROW_IE_EXCEPTION << "unexpected value " << quantizedTensorAlignmentOnActivations;
                }
                }

                if (updatePrecisions) {
                    CNNNetworkHelper::setOutDataPrecision(fakeQuantizeLayer, dataPrecision.precision);
                }

                lowNewOutput.push_back(dataPrecision.min * outputInterval[index] / maxOutputInterval);

                context.quantizedFakeQuantizeNames.insert(quantizeLayers[index]->name);
            }

            float generalScaleDequantize = maxOutputInterval / (dataPrecision.max - dataPrecision.min);
            const float quantizationShift =
                (sumLowOldOutput - generalScaleDequantize * accumulate(lowNewOutput.begin(), lowNewOutput.end(), 0.0)) *
                -1;
            dequantizationScales[channel] = generalScaleDequantize;
            dequantizationShifts[channel] = -1 * quantizationShift;
            for (int index = 0; index < numberInputLayers; index++) {
                dequantizationShiftsLayers[index].push_back((quantizationLayersDetails[index].outputLowValues[0] -
                                                             generalScaleDequantize * lowNewOutput[index]));
            }
        }
    }

    for (int index = 0; index < numberInputLayers; index++) {
        if (quantizeLayers[index]->outData[0]->getInputTo().size() != 1) {
            std::vector<CNNLayerPtr> children =
                CNNNetworkHelper::getChildren(*quantizeLayers[index], childNameOurAfterQuantizeLayers[index]);
            for (int i = 0; i < children.size(); i++) {
                const size_t outputChannelsCount = CNNNetworkHelper::getInputChannelsCount(*quantizeLayers[index]);
                CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                    context, std::make_shared<CNNLayer>(*quantizeLayers[index]), children[i],
                    DequantizationDetails(dequantizationScales, dequantizationShiftsLayers[index],
                                          outputChannelsCount));
                context.dequantizationLayersNames.insert(dequantizationLayer->name);
            }
        }
    }
    // Add scaleshift at other outputs of the Quantize layer

    const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(eltwise);
    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(eltwise);
    if (children.size() == 0) {
        const std::string originalName = eltwise.name;
        CNNNetworkHelper::renameLayer(context.network, eltwise.name,
                                      eltwise.name + LayerTransformation::lastLayerPrefix);

        CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
            context, std::make_shared<CNNLayer>(eltwise), nullptr,
            DequantizationDetails(dequantizationScales, dequantizationShifts, outputChannelsCount), originalName);
        context.dequantizationLayersNames.insert(dequantizationLayer->name);
    } else {
        for (const CNNLayerPtr& child : children) {
            CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context, std::make_shared<CNNLayer>(eltwise), child,
                DequantizationDetails(dequantizationScales, dequantizationShifts, outputChannelsCount));
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
        }
    }
}

bool EltwiseTransformation::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
    return false;
}

bool EltwiseTransformation::isBroadcasted(const TensorDesc& tensorDesc) {
    const std::vector<size_t> dims = tensorDesc.getDims();
    const size_t channelIndex = dims.size() == 1 ? 0ul : (dims.size() == 2ul ? 1ul : 2ul);
    for (size_t dimension = channelIndex; dimension < dims.size(); ++dimension) {
        if (dims[dimension] != 1ul) {
            return false;
        }
    }

    return true;
}

bool EltwiseTransformation::isSupported(const TensorDesc& tensorDesc1, const TensorDesc& tensorDesc2) {
    if (tensorDesc1.getPrecision() != tensorDesc2.getPrecision()) {
        return false;
    }

    const std::vector<size_t> dims1 = tensorDesc1.getDims();
    const size_t channelsCount1 = dims1.size() == 1ul ? dims1[0] : dims1[1];
    const std::vector<size_t> dims2 = tensorDesc2.getDims();
    const size_t channelsCount2 = dims2.size() == 1ul ? dims2[0] : dims2[1];
    if ((channelsCount1 != channelsCount2) && (channelsCount1 != 1ul) && (channelsCount2 != 1ul)) {
        return false;
    }

    if (((dims1.size() == 2ul) && (channelsCount1 == 1ul)) ||
        ((dims2.size() == 2ul) && (channelsCount2 == 1ul))) {
        return true;
    }

    if ((dims1 == dims2) && (tensorDesc1.getLayout() != tensorDesc2.getLayout())) {
        return false;
    }

    if (dims1 == dims2) {
        return true;
    }

    if ((dims1.size() > 1ul) && (dims2.size() > 1ul)) {
        if (dims1[1] != dims2[1]) {
            return false;
        }

        const size_t dimensionsSize = std::min(dims1.size(), dims2.size());
        for (size_t dimension = 2ul; dimension < dimensionsSize; ++dimension) {
            if ((dims1[dimension] != dims2[dimension]) && (dims1[dimension] != 1ul) && (dims2[dimension] != 1ul)) {
                return false;
            }
        }
    }

    return true;
}

size_t EltwiseTransformation::getMinQuantizationLevels(const DataPrecision& dataPrecision, const std::vector<float>& outputIntervals) {
    size_t minLevels = std::numeric_limits<std::size_t>::max();
    const float maxOutputInterval = *std::max_element(outputIntervals.begin(), outputIntervals.end());
    for (int index = 0; index < outputIntervals.size(); index++) {
        const float outputIntervalMin = roundf(dataPrecision.min * outputIntervals[index] / maxOutputInterval);
        const float outputIntervalMax = roundf(dataPrecision.max * outputIntervals[index] / maxOutputInterval);
        const size_t levels = static_cast<size_t>(fabs(outputIntervalMax - outputIntervalMin)) + 1ul;
        if (minLevels > levels) {
            minLevels = levels;
        }
    }
    return minLevels;
}
