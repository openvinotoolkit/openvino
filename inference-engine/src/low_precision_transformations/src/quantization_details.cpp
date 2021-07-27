// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <low_precision/quantization_details.hpp>
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

#include "low_precision/lpt_itt.hpp"

#include <low_precision/common/ie_lpt_exception.hpp>
#include <low_precision/network_helper.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

QuantizationDetails::QuantizationDetails()
    : levels(),
      inputLowValues({}),
      inputHighValues({}),
      outputLowValues({}),
      outputHighValues({}) {}

QuantizationDetails::QuantizationDetails(const QuantizationDetails& quantizationDetails)
    : levels(quantizationDetails.levels),
      inputLowValues(quantizationDetails.inputLowValues),
      inputHighValues(quantizationDetails.inputHighValues),
      outputLowValues(quantizationDetails.outputLowValues),
      outputHighValues(quantizationDetails.outputHighValues) {}

QuantizationDetails::QuantizationDetails(const size_t levels, const std::vector<float>& inputLowValues,
                                         const std::vector<float>& inputHighValues,
                                         const std::vector<float>& outputLowValues,
                                         const std::vector<float>& outputHighValues)
    : levels(levels),
      inputLowValues(inputLowValues),
      inputHighValues(inputHighValues),
      outputLowValues(outputLowValues),
      outputHighValues(outputHighValues) {}

bool QuantizationDetails::outputLayoutIsSupported(std::shared_ptr<opset1::FakeQuantize> quantize) {
    return is_type<opset1::Constant>(quantize->get_input_node_ptr(1)) &&
        is_type<opset1::Constant>(quantize->get_input_node_ptr(2)) &&
        is_type<opset1::Constant>(quantize->get_input_node_ptr(3)) &&
        is_type<opset1::Constant>(quantize->get_input_node_ptr(4));
}

void QuantizationDetails::getInputIntervals(
        std::shared_ptr<opset1::FakeQuantize> quantize,
        std::vector<float>& inputLowValues,
        std::vector<float>& inputHighValues) {
    std::shared_ptr<opset1::Constant> inputLowLayer = as_type_ptr<opset1::Constant>(quantize->get_input_node_shared_ptr(1));
    const std::vector<float>& inputLowBlobValues = getBlobValue(inputLowLayer);
    inputLowValues.insert(inputLowValues.end(), inputLowBlobValues.begin(), inputLowBlobValues.end());

    std::shared_ptr<opset1::Constant> inputHighLayer = as_type_ptr<opset1::Constant>(quantize->get_input_node_shared_ptr(2));
    const std::vector<float> inputHighBlobValues = getBlobValue(inputHighLayer);
    inputHighValues.insert(inputHighValues.end(), inputHighBlobValues.begin(), inputHighBlobValues.end());

    if (inputLowValues.size() != inputHighValues.size()) {
        THROW_IE_LPT_EXCEPTION(*quantize) << "Quantize input values sizes are not equal for layer " << quantize->get_friendly_name();
    }
}


void QuantizationDetails::getOutputIntervals(
        std::shared_ptr<opset1::FakeQuantize> quantize,
        std::vector<float>& outputLowValues,
        std::vector<float>& outputHighValues) {
    std::shared_ptr<opset1::Constant> outputLowLayer = as_type_ptr<opset1::Constant>(quantize->get_input_node_shared_ptr(3));
    const std::vector<float>& outputLowBlobValues = getBlobValue(outputLowLayer);
    outputLowValues.insert(outputLowValues.end(), outputLowBlobValues.begin(), outputLowBlobValues.end());

    std::shared_ptr<opset1::Constant> outputHighLayer = as_type_ptr<opset1::Constant>(quantize->get_input_node_shared_ptr(4));
    const std::vector<float> outputHighBlobValues = getBlobValue(outputHighLayer);
    outputHighValues.insert(outputHighValues.end(), outputHighBlobValues.begin(), outputHighBlobValues.end());

    if (outputLowValues.size() != outputHighValues.size()) {
        THROW_IE_LPT_EXCEPTION(*quantize) << "Quantize output values sizes are not equal for layer " << quantize->get_friendly_name();
    }
}

QuantizationDetails QuantizationDetails::getDetails(std::shared_ptr<opset1::FakeQuantize> quantize) {
    const std::vector<float> inputLowValues = as_type_ptr<opset1::Constant>(quantize->get_input_node_shared_ptr(1))->cast_vector<float>();
    const std::vector<float> inputHighValues = as_type_ptr<opset1::Constant>(quantize->get_input_node_shared_ptr(2))->cast_vector<float>();

    const std::vector<float> outputLowValues = as_type_ptr<opset1::Constant>(quantize->get_input_node_shared_ptr(3))->cast_vector<float>();
    const std::vector<float> outputHighValues = as_type_ptr<opset1::Constant>(quantize->get_input_node_shared_ptr(4))->cast_vector<float>();

    return QuantizationDetails(
        quantize->get_levels(),
        inputLowValues,
        inputHighValues,
        outputLowValues,
        outputHighValues);
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

float QuantizationDetails::getInputLowValue(const size_t index) const {
    return inputLowValues.size() == 1ul ? inputLowValues[0] : inputLowValues[index];
}

float QuantizationDetails::getInputHighValue(const size_t index) const {
    return inputHighValues.size() == 1ul ? inputHighValues[0] : inputHighValues[index];
}

float QuantizationDetails::getOutputLowValue(const size_t index) const {
    return outputLowValues.size() == 1ul ? outputLowValues[0] : outputLowValues[index];
}

float QuantizationDetails::getOutputHighValue(const size_t index) const {
    return outputHighValues.size() == 1ul ? outputHighValues[0] : outputHighValues[index];
}

std::vector<float> QuantizationDetails::getBlobValue(std::shared_ptr<Node> constantLayer) {
    return as_type_ptr<opset1::Constant>(constantLayer)->cast_vector<float>();
}

bool QuantizationDetails::isSupportedLevel(const size_t level) {
    static const std::unordered_set<size_t> supported_levels = { 255ul, 256ul };
    return supported_levels.find(level) != supported_levels.end();
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
