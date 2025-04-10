// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/mat_mul_with_optimized_constant_fake_quantize.hpp"

#include "openvino/opsets/opset1.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> MatMulWithOptimizedConstantFakeQuantizeFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape1,
    const ov::PartialShape& inputShape2,
    const FakeQuantizeOnData& fqOnData,
    const FakeQuantizeOnData& fqOnWeights) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape1);

    const auto lowConstantOnActivations = std::make_shared<ov::opset1::Constant>(precision, fqOnData.constantShape, fqOnData.inputLowValues);
    const auto highConstantOnActivations = std::make_shared<ov::opset1::Constant>(precision, fqOnData.constantShape, fqOnData.inputHighValues);
    const auto fakeQuantizeOnActivations = std::make_shared<ov::opset1::FakeQuantize>(
        input,
        lowConstantOnActivations,
        highConstantOnActivations,
        lowConstantOnActivations,
        highConstantOnActivations,
        fqOnWeights.quantizationLevel);

    const ov::Shape weightsShape = { static_cast<size_t>(inputShape2[0].get_length()), static_cast<size_t>(inputShape1[1].get_length()) };
    const std::vector<float> weigths(weightsShape[0] * weightsShape[1], 10.f);

    const auto weightsConst = std::make_shared<ov::opset1::Constant>(precision, weightsShape, weigths);
    const auto lowConstantOnWeights = std::make_shared<ov::opset1::Constant>(precision, fqOnWeights.constantShape, fqOnWeights.inputLowValues);
    const auto highConstantOnWeights = std::make_shared<ov::opset1::Constant>(precision, fqOnWeights.constantShape, fqOnWeights.inputHighValues);
    const auto fakeQuantizeOnWeights = std::make_shared<ov::opset1::FakeQuantize>(
        weightsConst,
        lowConstantOnWeights,
        highConstantOnWeights,
        lowConstantOnWeights,
        highConstantOnWeights,
        fqOnWeights.quantizationLevel);

    const auto matMul = std::make_shared<ov::opset1::MatMul>(
        fakeQuantizeOnActivations,
        fakeQuantizeOnWeights,
        false,
        inputShape1[1] != inputShape2[0]);

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(matMul) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "MatMulWithOptimizedConstantFakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
