// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fusing_test_utils.hpp"

using namespace LayerTestsDefinitions;

namespace FusingTestUtils {

std::shared_ptr<ngraph::Function> makeActivationPattern(std::vector<size_t> shape, ngraph::helpers::ActivationTypes type,
                                                        double alpha, double beta) {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(InferenceEngine::Precision::FP32);
    auto params = ngraph::builder::makeParams(ngPrc, {shape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto postOp = ngraph::builder::makeActivation(paramOuts[0], ngPrc, type, alpha, beta);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(postOp)};
    auto func = std::make_shared<ngraph::Function>(results, params, activationNames[type]);
    return func;
}

std::shared_ptr<ngraph::Function> makeSwishPattern(std::vector<size_t> shape) {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(InferenceEngine::Precision::FP32);
    auto params = ngraph::builder::makeParams(ngPrc, {shape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto sigmoid = ngraph::builder::makeActivation(paramOuts[0], ngPrc, ngraph::helpers::Sigmoid);
    auto multiply = std::make_shared<ngraph::opset1::Multiply>(paramOuts[0], sigmoid);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(multiply)};
    auto func = std::make_shared<ngraph::Function>(results, params, "SwishOptimization");
    return func;
}

std::shared_ptr<ngraph::Function> makeActivationScaleShiftPattern(ngraph::helpers::ActivationTypes type, std::vector<size_t> shape) {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(InferenceEngine::Precision::FP32);
    auto params = ngraph::builder::makeParams(ngPrc, {shape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto activation = ngraph::builder::makeActivation(paramOuts[0], ngPrc, type);
    auto scaleShift = ngraph::builder::makeScaleShift(activation, ngPrc, {}, {});
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(scaleShift)};
    auto func = std::make_shared<ngraph::Function>(results, params, "Activation_ScaleShift");
    return func;
}

std::shared_ptr<ngraph::Function> makeFakeQuantizeActivationPattern(size_t levels, ngraph::helpers::ActivationTypes type, std::vector<size_t> shape) {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(InferenceEngine::Precision::FP32);
    auto params = ngraph::builder::makeParams(ngPrc, {shape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto fq = ngraph::builder::makeFakeQuantize(paramOuts[0], ngPrc, levels, {1, 1, 1, 1});
    auto activation = ngraph::builder::makeActivation(fq, ngPrc, type);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(activation)};
    auto func = std::make_shared<ngraph::Function>(results, params, "FQ_Activation");
    return func;
}

std::shared_ptr<ngraph::Function> makeSumPattern(std::vector<size_t> shape) {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(InferenceEngine::Precision::FP32);
    auto params = ngraph::builder::makeParams(ngPrc, {shape, shape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto sum = std::make_shared<ngraph::op::Add>(paramOuts[0], paramOuts[1]);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(sum)};
    auto func = std::make_shared<ngraph::Function>(results, params, "Sum");
    return func;
}

} // namespace FusingTestUtils
