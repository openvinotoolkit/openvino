// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_weights.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class ActualValues {
public:
    ngraph::element::Type lowPrecision;
    std::vector<float> subtractValues;
    std::vector<float> mutliplyValues;
    std::vector<float> weightsValues;
    builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
};

class ExpectedValues {
public:
    ngraph::element::Type activationPrecision;
    std::vector<float> subtractValues;
    ngraph::element::Type weightsPrecision;
    std::vector<float> weightsValues;
    builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;;
    std::vector<float> mutliplyValues;
};

class ConvolutionFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ActualValues& actualValues);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ExpectedValues& expectedValues);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
