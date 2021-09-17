// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ostream>
#include <vector>

#include <low_precision/lpt_visibility.hpp>

#include <ngraph/node.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/type.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API QuantizationDetails {
public:
    QuantizationDetails();
    QuantizationDetails(const QuantizationDetails& quantizationDetails);
    QuantizationDetails(
            const size_t levels,
            const std::vector<float>& inputLowValues,
            const std::vector<float>& inputHighValues,
            const std::vector<float>& outputLowValues,
            const std::vector<float>& outputHighValues);

    static bool outputLayoutIsSupported(std::shared_ptr<opset1::FakeQuantize> quantize);

    static void getInputIntervals(
            std::shared_ptr<opset1::FakeQuantize> quantize,
            std::vector<float>& inputLowValues,
            std::vector<float>& inputHighValues);

    static void getOutputIntervals(
            std::shared_ptr<opset1::FakeQuantize> quantize,
            std::vector<float>& outputLowValues,
            std::vector<float>& outputHighValues);

    static QuantizationDetails getDetails(std::shared_ptr<opset1::FakeQuantize>);
    bool hasNegativeOutput() const;
    float maxOutput(const size_t channel) const;
    float maxInput(const size_t channel) const;

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

private:
    static std::vector<float> getBlobValue(std::shared_ptr<Node> constantLayer);
};

inline std::ostream &operator << (std::ostream &os, const QuantizationDetails& value) {
    os << "levels: " << value.levels <<
       ", input 1/" << value.inputLowValues.size() << ": [" << value.getInputLowValue(0) << " : " << value.getInputHighValue(0) << "], " <<
       ", output 1/" << value.outputLowValues.size() << ": [" << value.getOutputLowValue(0) << " : " << value.getOutputHighValue(0) << "]";
    return os;
}


} // namespace low_precision
} // namespace pass
} // namespace ngraph
