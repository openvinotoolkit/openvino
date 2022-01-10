// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/include/fuse_muladd_ewsimple.hpp"
#include "ngraph_functions/builders.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using ngraph::helpers::EltwiseTypes;
using ngraph::helpers::ActivationTypes;

namespace SubgraphTestsDefinitions {

std::string FuseMulAddAndEwSimpleTest::getTestCaseName(testing::TestParamInfo<FuseMulAddAndEwSimpleParams> obj) {
    std::ostringstream result;
    SizeVector inputShape;
    Precision inPrec;
    std::tie(inputShape, inPrec) = obj.param;

    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "Precision=" << inPrec.name();

    return result.str();
}

void FuseMulAddAndEwSimpleTest::SetUp() {
    targetDevice = CommonTestUtils::DEVICE_CPU;

    std::tie(inputShape, inPrec) = this->GetParam();
    CreateGraph();
}

const auto mulAddAndEwSimpleCommonParams = ::testing::Combine(
        ::testing::Values(SizeVector{1, 20}),
        ::testing::Values(Precision::FP32)
);


// Fused EltwiseAndSimple comes on the 3rd port into MulAdd
void FuseMulAddAndEwSimpleTest1::CreateGraph() {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);
    auto mulSecondInput = inputShape;
    mulSecondInput[0] = 1;
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape, inputShape, mulSecondInput});

    auto clamp = ngraph::builder::makeActivation(params[0], ngPrc, ActivationTypes::Clamp, inputShape, {0, 100});
    auto tanh = ngraph::builder::makeActivation(clamp, ngPrc, ActivationTypes::Tanh);
    auto mul1 = ngraph::builder::makeEltwise(params[1], params[2], EltwiseTypes::MULTIPLY);
    auto add = ngraph::builder::makeEltwise(tanh, mul1, EltwiseTypes::ADD);

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(add)};
    function = std::make_shared<ngraph::Function>(results, params, "MulAdd_EwSimple");
}

TEST_P(FuseMulAddAndEwSimpleTest1, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_Basic, FuseMulAddAndEwSimpleTest1, mulAddAndEwSimpleCommonParams, FuseMulAddAndEwSimpleTest::getTestCaseName);


// Fused EltwiseAndSimple comes on the 2nd input into MulAdd
void FuseMulAddAndEwSimpleTest2::CreateGraph() {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape, inputShape, inputShape});

    auto clamp1 = ngraph::builder::makeActivation(params[0], ngPrc, ActivationTypes::Clamp, inputShape, {0, 100});
    auto tanh1 = ngraph::builder::makeActivation(clamp1, ngPrc, ActivationTypes::Tanh);
    auto clamp2 = ngraph::builder::makeActivation(params[1], ngPrc, ActivationTypes::Clamp, inputShape, {0, 100});
    auto tanh2 = ngraph::builder::makeActivation(clamp2, ngPrc, ActivationTypes::Tanh);
    auto mul1 = ngraph::builder::makeEltwise(tanh2, tanh1, EltwiseTypes::MULTIPLY);
    auto add = ngraph::builder::makeEltwise(mul1, params[2], EltwiseTypes::ADD);

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(add)};
    function = std::make_shared<ngraph::Function>(results, params, "MulAdd_EwSimple_2");
}

TEST_P(FuseMulAddAndEwSimpleTest2, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_Basic, FuseMulAddAndEwSimpleTest2, mulAddAndEwSimpleCommonParams, FuseMulAddAndEwSimpleTest::getTestCaseName);


// Fused MulAdd with more than 3 inputs
void FuseMulAddAndEwSimpleTest3::CreateGraph() {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape, inputShape, inputShape, inputShape, inputShape});

    auto mul1 = ngraph::builder::makeEltwise(params[0], params[1], EltwiseTypes::MULTIPLY);
    auto add1 = ngraph::builder::makeEltwise(mul1, params[2], EltwiseTypes::ADD);
    auto tanh1 = ngraph::builder::makeActivation(add1, ngPrc, ActivationTypes::Tanh);
    auto mul2 = ngraph::builder::makeEltwise(tanh1, params[3], EltwiseTypes::MULTIPLY);
    auto add2 = ngraph::builder::makeEltwise(params[4], mul2, EltwiseTypes::ADD);

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(add2)};
    function = std::make_shared<ngraph::Function>(results, params, "MulAdd_EwSimple_3");
}

TEST_P(FuseMulAddAndEwSimpleTest3, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_Basic, FuseMulAddAndEwSimpleTest3, mulAddAndEwSimpleCommonParams, FuseMulAddAndEwSimpleTest::getTestCaseName);
}  // namespace SubgraphTestsDefinitions
