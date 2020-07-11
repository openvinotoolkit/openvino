// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/low_precision/quantization_details.hpp>
#include <math.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <transformations/low_precision/common/ie_lpt_exception.hpp>
#include <transformations/low_precision/network_helper.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

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

bool QuantizationDetails::outputLayoutIsSupported(std::shared_ptr<opset1::FakeQuantize> quantize) {
    std::vector<float> outputLowValues;
    std::vector<float> outputHighValues;
    size_t outputIntervalsCount;
    getOutputIntervals(quantize, outputLowValues, outputHighValues, outputIntervalsCount);

    // TODO: FQ on weights - temporary workaround:
    // if (outputIntervalsCount == quantize->get_output_shape(0)[0]) {
    //    return true;
    // }

    const size_t outputChannelsCount = NetworkHelper::getOutputChannelsCount(quantize, NetworkHelper::onWeights(quantize));
    if ((outputIntervalsCount != 1ul) && (outputIntervalsCount != outputChannelsCount)) {
        return false;
    }

    return true;
}

void QuantizationDetails::getInputIntervals(
        std::shared_ptr<opset1::FakeQuantize> quantize,
        std::vector<float>& inputLowValues,
        std::vector<float>& inputHighValues,
        size_t& inputIntervalsCount) {
    std::shared_ptr<opset1::Constant> inputLowLayer = as_type_ptr<opset1::Constant>(quantize->get_input_node_shared_ptr(1));
    validate(inputLowLayer);
    const std::vector<float>& inputLowBlobValues = getBlobValue(inputLowLayer);
    inputLowValues.insert(inputLowValues.end(), inputLowBlobValues.begin(), inputLowBlobValues.end());

    std::shared_ptr<opset1::Constant> inputHighLayer = as_type_ptr<opset1::Constant>(quantize->get_input_node_shared_ptr(2));
    validate(inputHighLayer);
    const std::vector<float> inputHighBlobValues = getBlobValue(inputHighLayer);
    inputHighValues.insert(inputHighValues.end(), inputHighBlobValues.begin(), inputHighBlobValues.end());

    if (inputLowValues.size() != inputHighValues.size()) {
        THROW_IE_LPT_EXCEPTION(*quantize) << "Quantize input values sizes are not equal for layer " << quantize->get_friendly_name();
    }

    inputIntervalsCount = inputLowValues.size();
}


void QuantizationDetails::getOutputIntervals(
        std::shared_ptr<opset1::FakeQuantize> quantize,
        std::vector<float>& outputLowValues,
        std::vector<float>& outputHighValues,
        size_t& outputIntervalsCount) {
    std::shared_ptr<opset1::Constant> outputLowLayer = as_type_ptr<opset1::Constant>(quantize->get_input_node_shared_ptr(3));
    validate(outputLowLayer);
    const std::vector<float>& outputLowBlobValues = getBlobValue(outputLowLayer);
    outputLowValues.insert(outputLowValues.end(), outputLowBlobValues.begin(), outputLowBlobValues.end());

    std::shared_ptr<opset1::Constant> outputHighLayer = as_type_ptr<opset1::Constant>(quantize->get_input_node_shared_ptr(4));
    validate(outputHighLayer);
    const std::vector<float> outputHighBlobValues = getBlobValue(outputHighLayer);
    outputHighValues.insert(outputHighValues.end(), outputHighBlobValues.begin(), outputHighBlobValues.end());

    if (outputLowValues.size() != outputHighValues.size()) {
        THROW_IE_LPT_EXCEPTION(*quantize) << "Quantize output values sizes are not equal for layer " << quantize->get_friendly_name();
    }

    outputIntervalsCount = outputLowValues.size();
}


QuantizationDetails QuantizationDetails::getDetails(std::shared_ptr<opset1::FakeQuantize> quantize) {
    std::vector<float> inputLowValues;
    std::vector<float> inputHighValues;
    size_t inputIntervalsCount;
    getInputIntervals(quantize, inputLowValues, inputHighValues, inputIntervalsCount);

    std::vector<float> outputLowValues;
    std::vector<float> outputHighValues;
    size_t outputIntervalsCount;
    getOutputIntervals(quantize, outputLowValues, outputHighValues, outputIntervalsCount);

    const size_t outputChannelsCount = NetworkHelper::getOutputChannelsCount(quantize, NetworkHelper::onWeights(quantize));
    if (!outputLayoutIsSupported(quantize)) {
        THROW_IE_LPT_EXCEPTION(*quantize) << "Expected output channels count " << outputIntervalsCount << " but found " << outputChannelsCount;
    }

    return QuantizationDetails(
            quantize->get_levels(),
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
        THROW_TRANSFORMATION_EXCEPTION << "channel " << channel << " is out of bound, input channels count " << inputIntervalsCount;
    }
    const float value = inputLowValues.size() == 1 ? inputLowValues[0] : inputLowValues[channel];
    return value;
}

float QuantizationDetails::getInputHighValue(const size_t channel) const {
    if ((inputIntervalsCount != 1) && (channel >= inputIntervalsCount)) {
        THROW_TRANSFORMATION_EXCEPTION << "channel " << channel << " is out of bound, input channels count " << inputIntervalsCount;
    }
    const float value = inputHighValues.size() == 1 ? inputHighValues[0] : inputHighValues[channel];
    return value;
}

float QuantizationDetails::getOutputLowValue(const size_t channel) const {
    if ((outputIntervalsCount != 1) && (channel >= outputIntervalsCount)) {
        THROW_TRANSFORMATION_EXCEPTION << "channel " << channel << " is out of bound, output channels count "
                           << outputIntervalsCount;
    }
    const float value = outputLowValues.size() == 1 ? outputLowValues[0] : outputLowValues[channel];
    return value;
}

float QuantizationDetails::getOutputHighValue(const size_t channel) const {
    if ((outputIntervalsCount != 1) && (channel >= outputIntervalsCount)) {
        THROW_TRANSFORMATION_EXCEPTION << "channel " << channel << " is out of bound, output channels count "
                           << outputIntervalsCount;
    }
    const float value = outputHighValues.size() == 1 ? outputHighValues[0] : outputHighValues[channel];
    return value;
}

void QuantizationDetails::validate(std::shared_ptr<Node> constantLayer) {
    // nothing to validate
    // TODO: remove?
}

std::vector<float> QuantizationDetails::getBlobValue(std::shared_ptr<Node> constantLayer) {
    return as_type_ptr<opset1::Constant>(constantLayer)->cast_vector<float>();
}

bool QuantizationDetails::isSupportedLevel(const size_t level) {
    static const std::unordered_set<size_t> supported_levels = { 15ul, 16ul, 255ul, 256ul };
    return supported_levels.find(level) != supported_levels.end();
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
