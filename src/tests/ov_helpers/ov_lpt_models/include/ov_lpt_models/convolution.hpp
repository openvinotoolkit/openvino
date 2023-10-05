// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

#include <low_precision/common/quantization_granularity_restriction.hpp>

#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class ConvolutionFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type netPrecision,
        const ngraph::element::Type inputPrecision,
        const ngraph::PartialShape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationOnActivations,
        std::shared_ptr<ngraph::opset1::Constant> weights,
        const ngraph::builder::subgraph::FakeQuantizeOnWeights fqOnWeights,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationOnWeights = DequantizationOperations(),
        const bool fqOnWeightsTransposeOnData = false,
        const bool fqOnWeightsTransposeOnInputLow = false,
        const bool fqOnWeightsTransposeOnInputHigh = false,
        const bool fqOnWeightsTransposeOnOutputLow = false,
        const bool fqOnWeightsTransposeOnOutputHigh = false);

    static std::shared_ptr<ngraph::Function> getOriginalWithIncorrectWeights(
        const ngraph::Shape& inputShape,
        ngraph::element::Type precision,
        ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
        ngraph::builder::subgraph::DequantizationOperations dequantization,
        bool isCrorrect);

    static std::shared_ptr<ngraph::Function> getOriginalWithIncorrectWeights(
        const ngraph::PartialShape& inputShape,
        ngraph::element::Type precision,
        ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData,
        bool isCorrect);

    static std::shared_ptr<ngraph::Function> getReferenceWithIncorrectWeights(
        const ngraph::Shape& inputShape,
        ngraph::element::Type inputPrecision,
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore,
        ngraph::element::Type weightsPrecision,
        std::vector<float> weightsValues,
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type netPrecision,
        const ngraph::element::Type inputPrecision,
        const ngraph::PartialShape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        std::shared_ptr<ngraph::opset1::Constant> weights,
        const ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
        const ngraph::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
        const ngraph::element::Type precisionAfterDequantization);

    static std::shared_ptr<ngraph::Function> get(
        const ngraph::Shape& inputShape,
        const ngraph::element::Type precision,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fakeQuantizeOnData,
        const std::vector<float>& weightsValues,
        const ngraph::builder::subgraph::FakeQuantizeOnWeights& fakeQuantizeOnWeights,
        const std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>& restrictions = {});
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
