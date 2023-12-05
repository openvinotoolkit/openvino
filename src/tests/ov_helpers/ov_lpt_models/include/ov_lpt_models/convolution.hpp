// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <low_precision/common/quantization_granularity_restriction.hpp>

#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class ConvolutionFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type netPrecision,
        const ov::element::Type inputPrecision,
        const ov::PartialShape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationOnActivations,
        std::shared_ptr<ov::opset1::Constant> weights,
        const ngraph::builder::subgraph::FakeQuantizeOnWeights fqOnWeights,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationOnWeights = DequantizationOperations(),
        const bool fqOnWeightsTransposeOnData = false,
        const bool fqOnWeightsTransposeOnInputLow = false,
        const bool fqOnWeightsTransposeOnInputHigh = false,
        const bool fqOnWeightsTransposeOnOutputLow = false,
        const bool fqOnWeightsTransposeOnOutputHigh = false);

    static std::shared_ptr<ov::Model> getOriginalWithIncorrectWeights(
        const ov::Shape& inputShape,
        ov::element::Type precision,
        ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
        ngraph::builder::subgraph::DequantizationOperations dequantization,
        bool isCrorrect);

    static std::shared_ptr<ov::Model> getOriginalWithIncorrectWeights(
        const ov::PartialShape& inputShape,
        ov::element::Type precision,
        ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
        ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData,
        bool isCorrect);

    static std::shared_ptr<ov::Model> getReferenceWithIncorrectWeights(
        const ov::Shape& inputShape,
        ov::element::Type inputPrecision,
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore,
        ov::element::Type weightsPrecision,
        std::vector<float> weightsValues,
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter);

    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type netPrecision,
        const ov::element::Type inputPrecision,
        const ov::PartialShape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        std::shared_ptr<ov::opset1::Constant> weights,
        const ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
        const ov::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
        const ov::element::Type precisionAfterDequantization);

    static std::shared_ptr<ov::Model> get(
        const ov::Shape& inputShape,
        const ov::element::Type precision,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fakeQuantizeOnData,
        const std::vector<float>& weightsValues,
        const ngraph::builder::subgraph::FakeQuantizeOnWeights& fakeQuantizeOnWeights,
        const std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>& restrictions = {});
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
