// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/include/fuse_muladd_ewsimple.hpp"
#include "ov_models/builders.hpp"
#include "ngraph/opsets/opset5.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using ov::helpers::EltwiseTypes;
using ov::helpers::ActivationTypes;

namespace SubgraphTestsDefinitions {

std::string FuseMulAddAndEwSimpleTest::getTestCaseName(testing::TestParamInfo<FuseMulAddAndEwSimpleParams> obj) {
    std::ostringstream result;
    SizeVector inputShape;
    Precision inPrec;
    std::tie(inputShape, inPrec) = obj.param;

    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "Precision=" << inPrec.name();

    return result.str();
}

void FuseMulAddAndEwSimpleTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

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
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape)),
                               std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape)),
                               std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(mulSecondInput))};

    auto clamp = ov::builder::makeActivation(params[0], ngPrc, ActivationTypes::Clamp, inputShape, {0, 100});
    auto tanh = ov::builder::makeActivation(clamp, ngPrc, ActivationTypes::Tanh);
    auto mul1 = ov::builder::makeEltwise(params[1], params[2], EltwiseTypes::MULTIPLY);
    auto add = ov::builder::makeEltwise(tanh, mul1, EltwiseTypes::ADD);

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(add)};
    function = std::make_shared<ngraph::Function>(results, params, "MulAdd_EwSimple");
}

TEST_P(FuseMulAddAndEwSimpleTest1, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_Basic, FuseMulAddAndEwSimpleTest1, mulAddAndEwSimpleCommonParams, FuseMulAddAndEwSimpleTest::getTestCaseName);


// Fused EltwiseAndSimple comes on the 2nd input into MulAdd
void FuseMulAddAndEwSimpleTest2::CreateGraph() {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape)),
                               std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape)),
                               std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    auto clamp1 = ov::builder::makeActivation(params[0], ngPrc, ActivationTypes::Clamp, inputShape, {0, 100});
    auto tanh1 = ov::builder::makeActivation(clamp1, ngPrc, ActivationTypes::Tanh);
    auto clamp2 = ov::builder::makeActivation(params[1], ngPrc, ActivationTypes::Clamp, inputShape, {0, 100});
    auto tanh2 = ov::builder::makeActivation(clamp2, ngPrc, ActivationTypes::Tanh);
    auto mul1 = ov::builder::makeEltwise(tanh2, tanh1, EltwiseTypes::MULTIPLY);
    auto add = ov::builder::makeEltwise(mul1, params[2], EltwiseTypes::ADD);

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(add)};
    function = std::make_shared<ngraph::Function>(results, params, "MulAdd_EwSimple_2");
}

TEST_P(FuseMulAddAndEwSimpleTest2, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_Basic, FuseMulAddAndEwSimpleTest2, mulAddAndEwSimpleCommonParams, FuseMulAddAndEwSimpleTest::getTestCaseName);


// Fused MulAdd with more than 3 inputs
void FuseMulAddAndEwSimpleTest3::CreateGraph() {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrec);
    ov::ParameterVector params;
    for (auto&& shape : {inputShape, inputShape, inputShape, inputShape, inputShape}) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(shape)));
    }

    auto mul1 = ov::builder::makeEltwise(params[0], params[1], EltwiseTypes::MULTIPLY);
    auto add1 = ov::builder::makeEltwise(mul1, params[2], EltwiseTypes::ADD);
    auto tanh1 = ov::builder::makeActivation(add1, ngPrc, ActivationTypes::Tanh);
    auto mul2 = ov::builder::makeEltwise(tanh1, params[3], EltwiseTypes::MULTIPLY);
    auto add2 = ov::builder::makeEltwise(params[4], mul2, EltwiseTypes::ADD);

    ngraph::ResultVector results{std::make_shared<ngraph::opset5::Result>(add2)};
    function = std::make_shared<ngraph::Function>(results, params, "MulAdd_EwSimple_3");
}

TEST_P(FuseMulAddAndEwSimpleTest3, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_Basic, FuseMulAddAndEwSimpleTest3, mulAddAndEwSimpleCommonParams, FuseMulAddAndEwSimpleTest::getTestCaseName);
}  // namespace SubgraphTestsDefinitions
