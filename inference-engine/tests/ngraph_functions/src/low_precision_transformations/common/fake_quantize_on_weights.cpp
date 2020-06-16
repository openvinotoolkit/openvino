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
    const std::vector<float>& lowValues,
    const std::vector<float>& highValues) : FakeQuantizeOnData(quantizationLevel, constantShape, lowValues, highValues) {}

FakeQuantizeOnWeights::~FakeQuantizeOnWeights() {}

bool FakeQuantizeOnWeights::empty() const {
    // TODO: add weights specific logic here
    return FakeQuantizeOnData::empty();
}
}
}
}
