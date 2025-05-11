// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/activation.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ActivationLayerTest;
using ov::test::ActivationParamLayerTest;
using ov::test::utils::ActivationTypes;

// Common params
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {ActivationTypes::Sigmoid,               {}},
        {ActivationTypes::Tanh,                  {}},
        {ActivationTypes::Relu,                  {}},
        {ActivationTypes::Exp,                   {}},
        {ActivationTypes::Log,                   {}},
        {ActivationTypes::Sign,                  {}},
        {ActivationTypes::Abs,                   {}},
        {ActivationTypes::Gelu,                  {}},
        {ActivationTypes::Clamp,                 {{-2.0f, 2.0f}}},
        {ActivationTypes::Negative,              {}},
        {ActivationTypes::Acos,                  {}},
        {ActivationTypes::Acosh,                 {}},
        {ActivationTypes::Asin,                  {}},
        {ActivationTypes::Asinh,                 {}},
        {ActivationTypes::Atan,                  {}},
        {ActivationTypes::Atanh,                 {}},
        {ActivationTypes::Cos,                   {}},
        {ActivationTypes::Cosh,                  {}},
        {ActivationTypes::Floor,                 {}},
        {ActivationTypes::Sin,                   {}},
        {ActivationTypes::Sinh,                  {}},
        {ActivationTypes::Sqrt,                  {}},
        {ActivationTypes::Tan,                   {}},
        {ActivationTypes::Elu,                   {{0.1f}}},
        {ActivationTypes::Erf,                   {}},
        {ActivationTypes::HardSigmoid,           {{0.2f, 0.5f}}},
        {ActivationTypes::Selu,                  {{1.6732f, 1.0507f}}},
        {ActivationTypes::Ceiling,               {}},
        {ActivationTypes::Mish,                  {}},
        {ActivationTypes::HSwish,                {}},
        {ActivationTypes::SoftPlus,              {}},
        {ActivationTypes::HSigmoid,              {}},
        {ActivationTypes::Swish,                 {{0.5f}}},
        {ActivationTypes::RoundHalfToEven,       {}},
        {ActivationTypes::RoundHalfAwayFromZero, {}},
        {ActivationTypes::GeluErf,               {}},
        {ActivationTypes::GeluTanh,              {}},
        {ActivationTypes::SoftSign,              {}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> big_rank_activation_types = {
        {ActivationTypes::Relu,                  {}},
        {ActivationTypes::Exp,                   {}},
        {ActivationTypes::Log,                   {}},
        {ActivationTypes::Abs,                   {}},
        {ActivationTypes::Clamp,                 {{-2.0f, 2.0f}}},
        {ActivationTypes::Ceiling,               {}},
        {ActivationTypes::Swish,                 {{0.5f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationParamTypes = {
        {ActivationTypes::PReLu, {{-0.01f}}},
        {ActivationTypes::LeakyRelu, {{0.01f}}}
};

std::map<std::vector<ov::Shape>, std::vector<ov::Shape>> basic = {
        {{{1, 50}}, {{}}},
        {{{1, 128}}, {{}}},
};

std::map<std::vector<ov::Shape>, std::vector<ov::Shape>> big_ranks = {
        {{{1, 2, 3, 4, 5, 3}}, {{}}},
        {{{1, 2, 3, 4, 1, 3, 2}}, {{}}},
        {{{1, 2, 3, 4, 3, 2, 1, 2}}, {{}}},
};

std::map<std::vector<ov::Shape>, std::vector<ov::Shape>> preluBasic = {
        {{{1, 10, 20}}, {{10}, {20}, {10, 20}}},
        {{{1, 128}}, {{1}, {128}}},
};

auto static_shapes_param_transform = [](const std::vector<std::pair<std::vector<ov::Shape>, ov::Shape>>& original_shapes) {
    std::vector<std::pair<std::vector<ov::test::InputShape>, ov::Shape>> new_shapes;
    for (const auto& shape_element : original_shapes) {
        new_shapes.emplace_back(ov::test::static_shapes_to_test_representation(shape_element.first), shape_element.second);
    }
    return new_shapes;
};

const auto basicCases = []() {
    return ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationTypes)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(static_shapes_param_transform(ov::test::utils::combineParams(basic))),
        ::testing::Values(ov::test::utils::DEVICE_GPU));
};

const auto basicPreluCases = []() {
    return ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationParamTypes)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(static_shapes_param_transform(ov::test::utils::combineParams(preluBasic))),
        ::testing::Values(ov::test::utils::DEVICE_GPU));
};

const auto big_rank_cases = []() {
    return ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(big_rank_activation_types)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(static_shapes_param_transform(ov::test::utils::combineParams(big_ranks))),
        ::testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic, ActivationLayerTest, basicCases(), ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Activation_BigRanks, ActivationLayerTest, big_rank_cases(), ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic_Prelu, ActivationLayerTest, basicPreluCases(), ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic, ActivationParamLayerTest, basicPreluCases(), ActivationLayerTest::getTestCaseName);

}  // namespace
