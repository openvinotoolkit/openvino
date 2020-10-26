// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "common/fake_quantize_on_data.hpp"
#include "low_precision/layer_transformation.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class MaxPoolFunction {
public:
    class ActualValues {
    public:
        ngraph::element::Type lowPrecision;
        std::vector<float> subtractValues;
        std::vector<float> mutliplyValues;
    };

    class ExpectedValues {
    public:
        ngraph::element::Type activationPrecision;
        std::vector<float> subtractValues;
        std::vector<float> mutliplyValues;
    };

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::Shape& inputShape,
        const ActualValues& values);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fakeQuantizeOnData);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::Shape& inputShape,
        const ExpectedValues& values);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
