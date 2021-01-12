// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

FakeQuantizeOnData::FakeQuantizeOnData() : quantizationLevel(0) {}

FakeQuantizeOnData::FakeQuantizeOnData(
    const size_t quantizationLevel,
    const ngraph::Shape& constantShape,
    const std::vector<float>& inputLowValues,
    const std::vector<float>& inputHighValues,
    const std::vector<float>& outputLowValues,
    const std::vector<float>& outputHighValues,
    const ngraph::element::Type outputPrecision) :
    quantizationLevel(quantizationLevel),
    constantShape(constantShape),
    inputLowValues(inputLowValues),
    inputHighValues(inputHighValues),
    outputLowValues(outputLowValues),
    outputHighValues(outputHighValues),
    outputPrecision(outputPrecision)
{}

FakeQuantizeOnData::~FakeQuantizeOnData() {}

bool FakeQuantizeOnData::isSigned() const {
    return std::any_of(outputLowValues.begin(), outputLowValues.end(), [](const float value) { return value < 0.f; }) ||
        std::any_of(outputHighValues.begin(), outputHighValues.end(), [](const float value) { return value < 0.f; });
}

bool FakeQuantizeOnData::empty() const {
    return (quantizationLevel == 0ul) &&
        constantShape.empty() &&
        inputLowValues.empty() &&
        inputHighValues.empty() &&
        outputLowValues.empty() &&
        outputHighValues.empty();
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
