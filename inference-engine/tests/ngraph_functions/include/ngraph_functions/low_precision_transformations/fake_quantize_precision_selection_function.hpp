// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <ngraph/ngraph.hpp>
#include "transformations/low_precision/layer_transformation.hpp"
#include "common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_weights.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class FakeQuantizePrecisionSelectionFunction {
public:
    class ActualValues {
    public:
        // bool isBeforeLimitPrecisionTransparentOperation;
        builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    };

    class ExpectedValues {
    public:
        ngraph::element::Type fakeQuantizeOnDataOutPrecision;
        builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
        builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    };

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ActualValues& values);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ExpectedValues& values);
};

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizePrecisionSelectionFunction::ActualValues& values) {
    return out << values.fakeQuantizeOnData << "_" << values.fakeQuantizeOnWeights;
}

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizePrecisionSelectionFunction::ExpectedValues& values) {
    return out << values.fakeQuantizeOnDataOutPrecision << "_" << values.fakeQuantizeOnData << "_" << values.fakeQuantizeOnWeights;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
