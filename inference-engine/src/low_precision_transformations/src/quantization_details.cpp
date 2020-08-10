// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/quantization_details.hpp"
#include <math.h>

#include <algorithm>
#include <blob_factory.hpp>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ie_common.h>
#include <legacy/cnn_network_impl.hpp>
#include <legacy/ie_util_internal.hpp>
#include "low_precision_transformations/common/ie_lpt_exception.hpp"
#include "low_precision_transformations/network_helper.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

class ConstTensorDesc {
public:
    static void validate(const Layout layout, const std::vector<size_t>& dims) {
        switch (layout) {
        case Layout::SCALAR: {
            if (dims.size() != 0) {
                THROW_IE_EXCEPTION << "unexpected dimensions size " << dims.size() << " for layout " << layout;
            }
            break;
        }
        case Layout::C: {
            if (dims.size() != 1) {
                THROW_IE_EXCEPTION << "unexpected dimensions size " << dims.size() << " for layout " << layout;
            }
            break;
        }
        case Layout::NCHW: {
            if (dims.size() != 4) {
                THROW_IE_EXCEPTION << "unexpected dimensions size " << dims.size() << " for layout " << layout;
            }
            break;
        }
        default: {
            THROW_IE_EXCEPTION << "unexpected layout " << layout;
        }
        }
    }

    static size_t getChannelsCount(const Layout layout, const std::vector<size_t>& dims) {
        switch (layout) {
        case Layout::SCALAR: {
            return 1;
        }
        case Layout::C: {
            return dims[0];
        }
        case Layout::NCHW: {
            return dims[1];
        }
        default: {
            THROW_IE_EXCEPTION << "unexpected layout " << layout;
        }
        }
    }
};

QuantizationDetails::QuantizationDetails()
    : levels(),
      inputLowValues({}),
      inputHighValues({}),
      outputLowValues({}),
      outputHighValues({}),
      inputIntervalsCount(0),
      outputIntervalsCount(0),
      outputChannelsCount(0) {}

QuantizationDetails::QuantizationDetails(const QuantizationDetails& quantizationDetails)
    : levels(quantizationDetails.levels),
      inputLowValues(quantizationDetails.inputLowValues),
      inputHighValues(quantizationDetails.inputHighValues),
      outputLowValues(quantizationDetails.outputLowValues),
      outputHighValues(quantizationDetails.outputHighValues),
      inputIntervalsCount(quantizationDetails.inputIntervalsCount),
      outputIntervalsCount(quantizationDetails.outputIntervalsCount),
      outputChannelsCount(quantizationDetails.outputChannelsCount) {}

QuantizationDetails::QuantizationDetails(const size_t levels, const std::vector<float>& inputLowValues,
                                         const std::vector<float>& inputHighValues,
                                         const std::vector<float>& outputLowValues,
                                         const std::vector<float>& outputHighValues, const size_t inputIntervalsCount,
                                         const size_t outputIntervalsCount, const size_t outputChannelsCount)
    : levels(levels),
      inputLowValues(inputLowValues),
      inputHighValues(inputHighValues),
      outputLowValues(outputLowValues),
      outputHighValues(outputHighValues),
      inputIntervalsCount(inputIntervalsCount),
      outputIntervalsCount(outputIntervalsCount),
      outputChannelsCount(outputChannelsCount) {}

bool QuantizationDetails::outputLayoutIsSupported(const CNNLayer& quantize) {
    std::vector<float> outputLowValues;
    std::vector<float> outputHighValues;
    size_t outputIntervalsCount;
    getOutputIntervals(quantize, outputLowValues, outputHighValues, outputIntervalsCount);

    const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(
        quantize,
        CNNNetworkHelper::onWeights(quantize) && CNNNetworkHelper::onConstWeightsPath(quantize));
    if ((outputIntervalsCount != 1ul) && (outputIntervalsCount != outputChannelsCount)) {
        return false;
    }

    return true;
}

void QuantizationDetails::getInputIntervals(
    const CNNLayer& quantize,
    std::vector<float>& inputLowValues,
    std::vector<float>& inputHighValues,
    size_t& inputIntervalsCount) {
    if (quantize.insData.size() != 5) {
        THROW_IE_LPT_EXCEPTION(quantize) << "Unexpected inputs size " << quantize.insData.size();
    }

    const DataPtr inputLowData = quantize.insData[1].lock();
    if (inputLowData == nullptr) {
        THROW_IE_LPT_EXCEPTION(quantize) << "input low data is absent";
    }
    const CNNLayerPtr inputLowLayer = getCreatorLayer(inputLowData).lock();
    validate(inputLowLayer);
    const std::vector<float> inputLowBlobValues = getBlobValue(inputLowLayer);
    inputLowValues.insert(inputLowValues.end(), inputLowBlobValues.begin(), inputLowBlobValues.end());

    const DataPtr inputHighData = quantize.insData[2].lock();
    if (inputHighData == nullptr) {
        THROW_IE_LPT_EXCEPTION(quantize) << "input high data is absent";
    }
    const CNNLayerPtr inputHighLayer = getCreatorLayer(inputHighData).lock();
    validate(inputHighLayer);
    const std::vector<float> inputHighBlobValues = getBlobValue(inputHighLayer);
    inputHighValues.insert(inputHighValues.end(), inputHighBlobValues.begin(), inputHighBlobValues.end());

    if (inputLowValues.size() != inputHighValues.size()) {
        THROW_IE_LPT_EXCEPTION(quantize) << "Quantize input values sizes are not equal for layer " << quantize.name;
    }

    inputIntervalsCount = inputLowValues.size();
}

void QuantizationDetails::getOutputIntervals(
    const CNNLayer& quantize,
    std::vector<float>& outputLowValues,
    std::vector<float>& outputHighValues,
    size_t& outputIntervalsCount) {
    if (quantize.insData.size() != 5) {
        THROW_IE_LPT_EXCEPTION(quantize) << "unexpected inputs size " << quantize.insData.size();
    }

    const DataPtr outputLowData = quantize.insData[3].lock();
    if (outputLowData == nullptr) {
        THROW_IE_LPT_EXCEPTION(quantize) << "output low data is absent";
    }
    const CNNLayerPtr outputLowLayer = getCreatorLayer(outputLowData).lock();
    validate(outputLowLayer);
    const std::vector<float>& outputLowBlobValues = getBlobValue(outputLowLayer);
    outputLowValues.insert(outputLowValues.end(), outputLowBlobValues.begin(), outputLowBlobValues.end());

    const DataPtr outputHighData = quantize.insData[4].lock();
    if (outputHighData == nullptr) {
        THROW_IE_LPT_EXCEPTION(quantize) << "output high data is absent";
    }
    const CNNLayerPtr outputHighLayer = getCreatorLayer(outputHighData).lock();
    validate(outputHighLayer);
    const std::vector<float> outputHighBlobValues = getBlobValue(outputHighLayer);
    outputHighValues.insert(outputHighValues.end(), outputHighBlobValues.begin(), outputHighBlobValues.end());

    if (outputLowValues.size() != outputHighValues.size()) {
        THROW_IE_LPT_EXCEPTION(quantize) << "Quantize output values sizes are not equal for layer " << quantize.name;
    }

    outputIntervalsCount = outputLowValues.size();
}

QuantizationDetails QuantizationDetails::getDetails(const CNNLayer& quantize) {
    std::vector<float> inputLowValues;
    std::vector<float> inputHighValues;
    size_t inputIntervalsCount;
    getInputIntervals(quantize, inputLowValues, inputHighValues, inputIntervalsCount);

    std::vector<float> outputLowValues;
    std::vector<float> outputHighValues;
    size_t outputIntervalsCount;
    getOutputIntervals(quantize, outputLowValues, outputHighValues, outputIntervalsCount);

    const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(
        quantize,
        CNNNetworkHelper::onWeights(quantize) && CNNNetworkHelper::onConstWeightsPath(quantize));
    if (!outputLayoutIsSupported(quantize)) {
        THROW_IE_LPT_EXCEPTION(quantize) << "Expected output channels count " << outputIntervalsCount << " but found " << outputChannelsCount;
    }

    if (!quantize.CheckParamPresence("levels")) {
        THROW_IE_LPT_EXCEPTION(quantize) << "Parameter 'levels' is absent";
    }

    return QuantizationDetails(
        quantize.GetParamAsInt("levels"),
        inputLowValues,
        inputHighValues,
        outputLowValues,
        outputHighValues,
        inputIntervalsCount,
        outputIntervalsCount,
        outputChannelsCount);
}

bool QuantizationDetails::hasNegativeOutput() const {
    for (const float value : outputLowValues) {
        if (value < 0.f) {
            return true;
        }
    }

    for (const float value : outputHighValues) {
        if (value < 0.f) {
            return true;
        }
    }

    return false;
}

float QuantizationDetails::maxOutput(const size_t channel) const {
    const auto value = fmax(fabs(outputLowValues[outputLowValues.size() == 1 ? 0 : channel]),
                            fabs(outputHighValues[outputHighValues.size() == 1 ? 0 : channel]));
    return value;
}

float QuantizationDetails::maxInput(const size_t channel) const {
    const auto value = fmax(fabs(outputLowValues[inputLowValues.size() == 1 ? 0 : channel]),
                            fabs(outputHighValues[inputHighValues.size() == 1 ? 0 : channel]));
    return value;
}

float QuantizationDetails::maxOutputHigh() const {
    float output = getOutputHighValue(0);
    for (size_t channel = 1; channel < outputIntervalsCount; ++channel) {
        if (output < getOutputHighValue(channel)) {
            output = getOutputHighValue(channel);
        }
    }
    return output;
}

float QuantizationDetails::minOutputLow() const {
    float output = getOutputLowValue(0);
    for (size_t channel = 1; channel < outputIntervalsCount; ++channel) {
        if (output > getOutputLowValue(channel)) {
            output = getOutputLowValue(channel);
        }
    }
    return output;
}

float QuantizationDetails::getInputLowValue(const size_t channel) const {
    if ((inputIntervalsCount != 1) && (channel >= inputIntervalsCount)) {
        THROW_IE_EXCEPTION << "channel " << channel << " is out of bound, input channels count " << inputIntervalsCount;
    }
    const float value = inputLowValues.size() == 1 ? inputLowValues[0] : inputLowValues[channel];
    return value;
}

float QuantizationDetails::getInputHighValue(const size_t channel) const {
    if ((inputIntervalsCount != 1) && (channel >= inputIntervalsCount)) {
        THROW_IE_EXCEPTION << "channel " << channel << " is out of bound, input channels count " << inputIntervalsCount;
    }
    const float value = inputHighValues.size() == 1 ? inputHighValues[0] : inputHighValues[channel];
    return value;
}

float QuantizationDetails::getOutputLowValue(const size_t channel) const {
    if ((outputIntervalsCount != 1) && (channel >= outputIntervalsCount)) {
        THROW_IE_EXCEPTION << "channel " << channel << " is out of bound, output channels count "
                           << outputIntervalsCount;
    }
    const float value = outputLowValues.size() == 1 ? outputLowValues[0] : outputLowValues[channel];
    return value;
}

float QuantizationDetails::getOutputHighValue(const size_t channel) const {
    if ((outputIntervalsCount != 1) && (channel >= outputIntervalsCount)) {
        THROW_IE_EXCEPTION << "channel " << channel << " is out of bound, output channels count "
                           << outputIntervalsCount;
    }
    const float value = outputHighValues.size() == 1 ? outputHighValues[0] : outputHighValues[channel];
    return value;
}

void QuantizationDetails::validate(const CNNLayerPtr& constantLayer) {
    if (constantLayer == nullptr) {
        THROW_IE_EXCEPTION << "Quantize layer input is absent";
    }

    if (constantLayer->blobs.size() == 0) {
        THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' doesn't have blobs";
    }

    if (constantLayer->blobs.size() > 1) {
        THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' has too much blobs";
    }

    const auto blob = constantLayer->blobs.begin()->second;
    // const auto byteSize = blob->byteSize();
    // if ((blob->getTensorDesc().getDims().size() != 0) &&
    //     (blob->getTensorDesc().getDims().size() != 1) &&
    //     (blob->getTensorDesc().getDims().size() != 4)) {
    //     THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' blob dimensions are not correct";
    // }

    const auto tensorDesc = blob->getTensorDesc();
    // if ((tensorDesc.getLayout() != Layout::SCALAR) &&
    //     (tensorDesc.getLayout() != Layout::C) &&
    //     ((tensorDesc.getLayout() != Layout::NCHW))) {
    //     THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' layout not correct";
    // }

    // const auto dims = tensorDesc.getDims();
    // if ((dims.size() != 0) && (dims.size() != 1) && (dims.size() != 4)) {
    //     THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' blob dimensions size " <<
    //     dims.size() << " not correct";
    // }

    // ConstTensorDesc::validate(tensorDesc.getLayout(), tensorDesc.getDims());
}

std::vector<float> QuantizationDetails::getBlobValue(const CNNLayerPtr& constantLayer) {
    if (constantLayer->blobs.empty()) {
        THROW_IE_LPT_EXCEPTION(*constantLayer) << "blobs are empty";
    }
    const auto blob = constantLayer->blobs.begin()->second;
    auto buffer = CNNNetworkHelper::getFloatData(blob);
    return std::vector<float>(buffer.get(), buffer.get() + blob->size());
}

bool QuantizationDetails::isSupportedLevel(const size_t level) {
    static const std::unordered_set<size_t> supported_levels = { 15ul, 16ul, 255ul, 256ul };
    return supported_levels.find(level) != supported_levels.end();
}
