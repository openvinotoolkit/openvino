// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/mat_mul_with_optimized_constant_fake_quantize_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> MatMulWithOptimizedConstantFakeQuantizeFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData,
    const FakeQuantizeOnData& fqOnWeights) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    const auto fakeQuantizeOnActivations = fqOnData.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input, precision, fqOnData.quantizationLevel, fqOnData.constantShape,
            fqOnData.inputLowValues, fqOnData.inputHighValues, fqOnData.outputLowValues, fqOnData.outputHighValues);

    const ngraph::Shape weightsShape = { inputShape[1], 10 };
    const std::vector<float> weigths(weightsShape[0] * weightsShape[1], 10.f);
    const auto weightsConst = std::make_shared<ngraph::opset1::Constant>(precision, weightsShape, weigths);
    const auto lowConstant = std::make_shared<ngraph::opset1::Constant>(precision, fqOnWeights.constantShape, fqOnWeights.inputLowValues);
    const auto highConstant = std::make_shared<ngraph::opset1::Constant>(precision, fqOnWeights.constantShape, fqOnWeights.inputHighValues);
    const auto fakeQuantizeOnWeights = std::make_shared<ngraph::opset1::FakeQuantize>(
        weightsConst,
        lowConstant,
        highConstant,
        lowConstant,
        highConstant,
        fqOnWeights.quantizationLevel);

    const auto matMul = std::make_shared<ngraph::opset1::MatMul>(
        fqOnData.empty() ? input : fakeQuantizeOnActivations,
        fakeQuantizeOnWeights,
        false,
        false);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(matMul) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MatMulWithOptimizedConstantFakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
