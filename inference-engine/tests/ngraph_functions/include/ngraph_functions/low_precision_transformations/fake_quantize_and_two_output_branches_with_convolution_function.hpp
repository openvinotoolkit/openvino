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

class FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction {
public:
    class ActualValues {
    public:
        FakeQuantizeOnData fqOnData;
        FakeQuantizeOnWeights fqOnWeights1;
        FakeQuantizeOnWeights fqOnWeights2;
    };

    class ExpectedValues {
    public:
        FakeQuantizeOnData fqOnData;
        FakeQuantizeOnWeights fqOnWeights1;
        std::vector<float> multiplay1Values;
        FakeQuantizeOnWeights fqOnWeights2;
        std::vector<float> multiplay2Values;
    };

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ActualValues& values);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::pass::low_precision::LayerTransformation::Params& params,
        const ExpectedValues& values);
};

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::ActualValues& values) {
    return out << "_" << values.fqOnData << "_" << values.fqOnWeights1 << "_" << values.fqOnWeights2;
}

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::ExpectedValues& values) {
    return out << "_" << values.fqOnData << "_" << values.fqOnWeights1 << "_" << values.fqOnWeights2;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
