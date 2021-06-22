// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class GroupConvolutionFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::Shape& outputShape,
        const size_t groupCount,
        const int groupCalculationDimention,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        std::shared_ptr<ngraph::opset1::Constant> weightsConst,
        const ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::Shape& outputShape,
        const size_t groupCount,
        const int groupCalculationDimention,
        const FakeQuantizeOnData& fakeQuantizeOnData,
        const FakeQuantizeOnWeights& fakeQuantizeOnWeights);

    static std::shared_ptr<ngraph::Function> get(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::Shape& outputShape,
        const size_t groupCount,
        const int calculatedDimention,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        std::shared_ptr<ngraph::opset1::Constant> weightsConst,
        const ngraph::builder::subgraph::FakeQuantizeOnWeights& fakeQuantizeOnWeights,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationOnWeights,
        const ngraph::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
        const ngraph::element::Type precisionAfterDequantization);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
