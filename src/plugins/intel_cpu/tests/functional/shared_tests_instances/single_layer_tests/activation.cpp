// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_enums.hpp"
#include "single_op_tests/activation.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ActivationLayerTest;
using ov::test::ActivationParamLayerTest;
using ov::test::utils::ActivationTypes;

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {ActivationTypes::Sigmoid,               {}},
        {ActivationTypes::Tan,                   {}},
        {ActivationTypes::Tanh,                  {}},
        {ActivationTypes::Relu,                  {}},
        {ActivationTypes::Exp,                   {}},
        {ActivationTypes::Log,                   {}},
        {ActivationTypes::Sign,                  {}},
        {ActivationTypes::Abs,                   {}},
        {ActivationTypes::Clamp,                 {{-2.0f, 2.0f}}},
        {ActivationTypes::Negative,              {}},
        {ActivationTypes::Acos,                  {}},
        {ActivationTypes::Acosh,                  {}},
        {ActivationTypes::Asin,                  {}},
        {ActivationTypes::Asinh,                 {}},
        {ActivationTypes::Atan,                  {}},
        {ActivationTypes::Atanh,                  {}},
        {ActivationTypes::Cos,                   {}},
        {ActivationTypes::Cosh,                  {}},
        {ActivationTypes::Floor,                 {}},
        {ActivationTypes::Sin,                   {}},
        {ActivationTypes::Sinh,                  {}},
        {ActivationTypes::Sqrt,                  {}},
        {ActivationTypes::Elu,                   {{0.1f}}},
        {ActivationTypes::Erf,                   {}},
        {ActivationTypes::HardSigmoid,           {{0.2f, 0.5f}}},
        {ActivationTypes::Selu,                  {{1.6732f, 1.0507f}}},
        {ActivationTypes::Ceiling,               {}},
        {ActivationTypes::Mish,                  {}},
        {ActivationTypes::HSwish,                {}},
        {ActivationTypes::SoftPlus,              {}},
        {ActivationTypes::HSigmoid,              {}},
        {ActivationTypes::RoundHalfToEven,       {}},
        {ActivationTypes::RoundHalfAwayFromZero, {}},
        {ActivationTypes::GeluErf,               {}},
        {ActivationTypes::GeluTanh,              {}},
        {ActivationTypes::Swish,                 {{0.4f}}},
        {ActivationTypes::IsFinite,              {}},
        {ActivationTypes::IsInf,                 {}},
        {ActivationTypes::IsNaN,                 {{}}},
        {ActivationTypes::LogicalNot,            {}},
};

// List of operations that should be tested also with integer precision
const std::map<ActivationTypes, std::vector<std::vector<float>>> intActivationTypes = {
        {ActivationTypes::Acosh,                 {}},
        {ActivationTypes::Asinh,                 {}},
        {ActivationTypes::Atan,                  {}},
        {ActivationTypes::Negative,              {}},
        {ActivationTypes::Ceiling,               {}},
        {ActivationTypes::Cos,                   {}},
        {ActivationTypes::Cosh,                  {}},
        {ActivationTypes::Sign,                  {}},
        {ActivationTypes::Sinh,                  {}},
        {ActivationTypes::Sqrt,                  {}},
        {ActivationTypes::Tan,                   {}},
        {ActivationTypes::Tanh,                  {}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationParamTypes = {
        {ActivationTypes::PReLu, {{}}}, // Slope will be filled with increasing values from -10 to match slope input shape
        {ActivationTypes::LeakyRelu, {{0.01f}}}
};

std::map<std::vector<ov::Shape>, std::vector<ov::Shape>> basic_input_shapes_static = {
        {{{1, 50}}, {}},
        {{{5, 128}}, {}},
        {{{2, 2, 2, 2, 2, 2, 2, 2}}, {}},
};

std::map<std::vector<ov::Shape>, std::vector<ov::Shape>> prelu_basic_input_shapes_static = {
        {{{1, 50}}, {{1}, {50}}},
        {{{1, 128}}, {{1}, {128}}},

        // Broadcast check
        {{{3, 2}}, {{1}, {2}, {3, 2}}},
        {{{3, 2, 5}}, {{1}, {2}, {5}, {2, 5}, {3, 1, 5}, {1, 2, 1}, {1, 1, 5}, {3, 1, 1}, {3, 2, 5}}},
        {{{2, 1, 2}}, {{2}, {2, 1, 1}}},
        {{{3, 2, 5, 7}}, {{1}, {7}, {2}, {5, 7}, {2, 5, 7}, {2, 1, 1}, {1, 2, 1, 1}, {3, 2, 1, 1}, {3, 2, 5, 7}}},
        {{{2, 2, 2, 2, 2, 2, 2, 2}}, {{2}, {2, 2}, {2, 1, 1, 2}}},
};

auto static_shapes_param_transform = [](const std::vector<std::pair<std::vector<ov::Shape>, ov::Shape>>& original_shapes) {
    std::vector<std::pair<std::vector<ov::test::InputShape>, ov::Shape>> new_shapes;
    for (const auto& shape_element : original_shapes) {
        new_shapes.emplace_back(ov::test::static_shapes_to_test_representation(shape_element.first), shape_element.second);
    }
    return new_shapes;
};

const auto basic_case_params = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationTypes)),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(static_shapes_param_transform(ov::test::utils::combineParams(basic_input_shapes_static))),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto basic_prelu_cases_params = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationParamTypes)),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(static_shapes_param_transform(ov::test::utils::combineParams(prelu_basic_input_shapes_static))),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto basic_integer_operations_params = ::testing::Combine(
            ::testing::ValuesIn(ov::test::utils::combineParams(intActivationTypes)),
            ::testing::Values(ov::element::i32),
            ::testing::ValuesIn(static_shapes_param_transform(ov::test::utils::combineParams(basic_input_shapes_static))),
            ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic, ActivationLayerTest, basic_case_params, ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Integer_Activation_Basic, ActivationLayerTest, basic_integer_operations_params, ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic_Prelu_Const, ActivationLayerTest, basic_prelu_cases_params, ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic_Prelu_Param, ActivationParamLayerTest, basic_prelu_cases_params, ActivationLayerTest::getTestCaseName);
}  // namespace
