// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include <transformations_visibility.hpp>

#include <ngraph/node.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/type.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API QuantizationDetails {
public:
    QuantizationDetails();
    QuantizationDetails(const QuantizationDetails& quantizationDetails);
    QuantizationDetails(
            const size_t levels,
            const std::vector<float>& inputLowValues,
            const std::vector<float>& inputHighValues,
            const std::vector<float>& outputLowValues,
            const std::vector<float>& outputHighValues,
            const size_t inputIntervalsCount,
            const size_t outputIntervalsCount,
            const size_t outputChannelsCount);

    static bool outputLayoutIsSupported(std::shared_ptr<opset1::FakeQuantize> quantize);

    static void getInputIntervals(
            std::shared_ptr<opset1::FakeQuantize> quantize,
            std::vector<float>& inputLowValues,
            std::vector<float>& inputHighValues,
            size_t& inputIntervalsCount);

    static void getOutputIntervals(
            std::shared_ptr<opset1::FakeQuantize> quantize,
            std::vector<float>& outputLowValues,
            std::vector<float>& outputHighValues,
            size_t& outputIntervalsCount);

    static QuantizationDetails getDetails(std::shared_ptr<opset1::FakeQuantize>);
    bool hasNegativeOutput() const;
    float maxOutput(const size_t channel) const;
    float maxInput(const size_t channel) const;

    float maxOutputHigh() const;
    float minOutputLow() const;

    float getInputLowValue(const size_t channel) const;
    float getInputHighValue(const size_t channel) const;
    float getOutputLowValue(const size_t channel) const;
    float getOutputHighValue(const size_t channel) const;

    static bool isSupportedLevel(const size_t level);

    const size_t levels;
    const std::vector<float> inputLowValues;
    const std::vector<float> inputHighValues;
    const std::vector<float> outputLowValues;
    const std::vector<float> outputHighValues;
    const size_t inputIntervalsCount;
    const size_t outputIntervalsCount;
    const size_t outputChannelsCount;

private:
    QuantizationDetails &operator=(const QuantizationDetails & /*target*/) { return *this; }
    static void validate(std::shared_ptr<Node> constantLayer);
    static std::vector<float> getBlobValue(std::shared_ptr<Node> constantLayer);
};

inline std::ostream &operator << (std::ostream &os, const QuantizationDetails& value) {
    os << "levels: " << value.levels <<
       ", input 1/" << value.inputIntervalsCount << ": [" << value.getInputLowValue(0) << " : " << value.getInputHighValue(0) << "], " <<
       ", output 1/" << value.outputIntervalsCount << ": [" << value.getOutputLowValue(0) << " : " << value.getOutputHighValue(0) << "]";
    return os;
}


}// namespace low_precision
}// namespace pass
}// namespace ngraph