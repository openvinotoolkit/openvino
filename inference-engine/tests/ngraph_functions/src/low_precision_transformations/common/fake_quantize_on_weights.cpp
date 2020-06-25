// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_weights.hpp"
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

FakeQuantizeOnWeights::FakeQuantizeOnWeights() {}

FakeQuantizeOnWeights::FakeQuantizeOnWeights(
    const size_t quantizationLevel,
    const ngraph::Shape& constantShape,
    const std::vector<float>& inputLowValues,
    const std::vector<float>& inputHighValues,
    const std::vector<float>& outputLowValues,
    const std::vector<float>& outputHighValues) :
    FakeQuantizeOnData(quantizationLevel, constantShape, inputLowValues, inputHighValues, outputLowValues, outputHighValues) {}

FakeQuantizeOnWeights::~FakeQuantizeOnWeights() {}

bool FakeQuantizeOnWeights::empty() const {
    // TODO: add weights specific logic here
    return FakeQuantizeOnData::empty();
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
