// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/include/fuse_muladd_ewsimple.hpp"

#include "common_test_utils/node_builders/activation.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"

using namespace CPUTestUtils;
using ov::test::utils::ActivationTypes;
using ov::test::utils::EltwiseTypes;

namespace ov {
namespace test {

std::string FuseMulAddAndEwSimpleTest::getTestCaseName(testing::TestParamInfo<FuseMulAddAndEwSimpleParams> obj) {
    std::ostringstream result;
    ov::Shape inputShape;
    ov::element::Type inPrec;
    std::tie(inputShape, inPrec) = obj.param;

    result << "IS=" << inputShape << "_";
    result << "Precision=" << inPrec.get_type_name();

    return result.str();
}

void FuseMulAddAndEwSimpleTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    std::tie(inputShape, inPrec) = this->GetParam();
    CreateGraph();
}

const auto mulAddAndEwSimpleCommonParams =
    ::testing::Combine(::testing::Values(ov::Shape({1, 20})), ::testing::Values(ov::element::f32));

// Fused EltwiseAndSimple comes on the 3rd port into MulAdd
void FuseMulAddAndEwSimpleTest1::CreateGraph() {
    auto mulSecondInput = inputShape;
    mulSecondInput[0] = 1;
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inPrec, inputShape),
                               std::make_shared<ov::op::v0::Parameter>(inPrec, inputShape),
                               std::make_shared<ov::op::v0::Parameter>(inPrec, mulSecondInput)};

    auto clamp = ov::test::utils::make_activation(params[0], inPrec, ActivationTypes::Clamp, inputShape, {0, 100});
    auto tanh = ov::test::utils::make_activation(clamp, inPrec, ActivationTypes::Tanh);
    auto mul1 = ov::test::utils::make_eltwise(params[1], params[2], EltwiseTypes::MULTIPLY);
    auto add = ov::test::utils::make_eltwise(tanh, mul1, EltwiseTypes::ADD);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
    function = std::make_shared<ov::Model>(results, params, "MulAdd_EwSimple");
}

TEST_P(FuseMulAddAndEwSimpleTest1, CompareWithRefs) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_Basic,
                         FuseMulAddAndEwSimpleTest1,
                         mulAddAndEwSimpleCommonParams,
                         FuseMulAddAndEwSimpleTest::getTestCaseName);

// Fused EltwiseAndSimple comes on the 2nd input into MulAdd
void FuseMulAddAndEwSimpleTest2::CreateGraph() {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inPrec, inputShape),
                               std::make_shared<ov::op::v0::Parameter>(inPrec, inputShape),
                               std::make_shared<ov::op::v0::Parameter>(inPrec, inputShape)};

    auto clamp1 = ov::test::utils::make_activation(params[0], inPrec, ActivationTypes::Clamp, inputShape, {0, 100});
    auto tanh1 = ov::test::utils::make_activation(clamp1, inPrec, ActivationTypes::Tanh);
    auto clamp2 = ov::test::utils::make_activation(params[1], inPrec, ActivationTypes::Clamp, inputShape, {0, 100});
    auto tanh2 = ov::test::utils::make_activation(clamp2, inPrec, ActivationTypes::Tanh);
    auto mul1 = ov::test::utils::make_eltwise(tanh2, tanh1, EltwiseTypes::MULTIPLY);
    auto add = ov::test::utils::make_eltwise(mul1, params[2], EltwiseTypes::ADD);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
    function = std::make_shared<ov::Model>(results, params, "MulAdd_EwSimple_2");
}

TEST_P(FuseMulAddAndEwSimpleTest2, CompareWithRefs) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_Basic,
                         FuseMulAddAndEwSimpleTest2,
                         mulAddAndEwSimpleCommonParams,
                         FuseMulAddAndEwSimpleTest::getTestCaseName);

// Fused MulAdd with more than 3 inputs
void FuseMulAddAndEwSimpleTest3::CreateGraph() {
    ov::ParameterVector params;
    for (auto&& shape : {inputShape, inputShape, inputShape, inputShape, inputShape}) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(inPrec, shape));
    }

    auto mul1 = ov::test::utils::make_eltwise(params[0], params[1], EltwiseTypes::MULTIPLY);
    auto add1 = ov::test::utils::make_eltwise(mul1, params[2], EltwiseTypes::ADD);
    auto tanh1 = ov::test::utils::make_activation(add1, inPrec, ActivationTypes::Tanh);
    auto mul2 = ov::test::utils::make_eltwise(tanh1, params[3], EltwiseTypes::MULTIPLY);
    auto add2 = ov::test::utils::make_eltwise(params[4], mul2, EltwiseTypes::ADD);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add2)};
    function = std::make_shared<ov::Model>(results, params, "MulAdd_EwSimple_3");
}

TEST_P(FuseMulAddAndEwSimpleTest3, CompareWithRefs) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_Basic,
                         FuseMulAddAndEwSimpleTest3,
                         mulAddAndEwSimpleCommonParams,
                         FuseMulAddAndEwSimpleTest::getTestCaseName);
}  // namespace test
}  // namespace ov
