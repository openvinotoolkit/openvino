// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/fake_quantize_function.hpp"
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

FakeQuantizeOnData::FakeQuantizeOnData() {};

FakeQuantizeOnData::FakeQuantizeOnData(
    const size_t quantizationLevel,
    const ngraph::Shape& constantShape,
    const std::vector<float>& lowValues,
    const std::vector<float>& highValues) :
    quantizationLevel(quantizationLevel),
    constantShape(constantShape),
    lowValues(lowValues),
    highValues(highValues)
{}

FakeQuantizeOnData::~FakeQuantizeOnData() {};

bool FakeQuantizeOnData::isSigned() const {
    return std::any_of(lowValues.begin(), lowValues.end(), [](const float value) { return value < 0.f; }) ||
        std::any_of(highValues.begin(), highValues.end(), [](const float value) { return value < 0.f; });
}

bool FakeQuantizeOnData::empty() const {
    return (quantizationLevel == 0ul) &&
        constantShape.empty() &&
        lowValues.empty() &&
        highValues.empty();
}

}
}
}
