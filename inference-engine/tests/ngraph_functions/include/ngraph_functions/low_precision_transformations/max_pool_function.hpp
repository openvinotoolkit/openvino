// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "transformations/low_precision/layer_transformation.hpp"

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
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::pass::low_precision::LayerTransformation::Params& params,
        const ActualValues& values);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::pass::low_precision::LayerTransformation::Params& params,
        const ExpectedValues& values);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
