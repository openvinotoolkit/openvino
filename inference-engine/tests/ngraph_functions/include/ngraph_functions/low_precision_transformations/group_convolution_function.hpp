// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_weights.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

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
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        std::shared_ptr<ngraph::opset1::Constant> weightsConst,
        const ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::Shape& outputShape,
        const size_t groupCount,
        const FakeQuantizeOnData& fakeQuantizeOnData,
        const FakeQuantizeOnWeights& fakeQuantizeOnWeights);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::Shape& outputShape,
        const size_t groupCount,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        std::shared_ptr<ngraph::opset1::Constant> weightsConst,
        const ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
        const ngraph::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
        const ngraph::element::Type precisionAfterDequantization);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
