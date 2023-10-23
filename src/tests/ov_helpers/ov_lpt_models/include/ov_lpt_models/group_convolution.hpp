// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

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
        const ngraph::PartialShape& inputShape,
        const ngraph::Shape& outputShape,
        const size_t groupCount,
        const int groupCalculationDimention,
        const FakeQuantizeOnData& fakeQuantizeOnData,
        const FakeQuantizeOnWeights& fakeQuantizeOnWeights,
        const bool addReshape = true,
        const bool addPrecisionPreserved = false);

    static std::shared_ptr<ngraph::Function> get(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const ngraph::PartialShape& outputShape,
        const size_t groupCount,
        const int calculatedDimention,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        std::shared_ptr<ngraph::opset1::Constant> weightsConst,
        const ngraph::builder::subgraph::FakeQuantizeOnWeights& fakeQuantizeOnWeights,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationOnWeights,
        const ngraph::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
        const ngraph::element::Type precisionAfterDequantization,
        const bool addReshape);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
