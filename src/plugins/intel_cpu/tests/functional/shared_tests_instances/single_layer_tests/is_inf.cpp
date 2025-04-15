// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/is_inf.hpp"

namespace {
using ov::test::IsInfLayerTest;

std::vector<std::vector<ov::test::InputShape>> input_shapes_static = {
        { {{}, {{2}}} },
        { {{}, {{2, 200}}} },
        { {{}, {{10, 200}}} },
        { {{}, {{1, 10, 100}}} },
        { {{}, {{4, 4, 16}}} },
        { {{}, {{1, 1, 1, 3}}} },
        { {{}, {{2, 17, 5, 4}}} },
        { {{}, {{2, 17, 5, 1}}} },
        { {{}, {{1, 2, 4}}} },
        { {{}, {{1, 4, 4}}} },
        { {{}, {{1, 4, 4, 1}}} },
        { {{}, {{16, 16, 16, 16, 16}}} },
        { {{}, {{16, 16, 16, 16, 1}}} },
        { {{}, {{16, 16, 16, 1, 16}}} },
        { {{}, {{16, 32, 1, 1, 1}}} },
        { {{}, {{1, 1, 1, 1, 1, 1, 3}}} },
        { {{}, {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}} }
};

std::vector<std::vector<ov::test::InputShape>> input_shapes_dynamic = {
        {{{ov::Dimension(1, 10), 200}, {{2, 200}, {1, 200}}}}
};

std::vector<ov::element::Type> model_types = {
        ov::element::f32
};

std::vector<bool> detect_negative = {
    true, false
};

std::vector<bool> detect_positive = {
    true, false
};

ov::AnyMap additional_config = {};

const auto is_inf_params = ::testing::Combine(
        ::testing::ValuesIn(input_shapes_static),
        ::testing::ValuesIn(detect_negative),
        ::testing::ValuesIn(detect_positive),
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config));

const auto is_inf_params_dynamic = ::testing::Combine(
        ::testing::ValuesIn(input_shapes_dynamic),
        ::testing::ValuesIn(detect_negative),
        ::testing::ValuesIn(detect_positive),
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_static, IsInfLayerTest, is_inf_params, IsInfLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_dynamic, IsInfLayerTest, is_inf_params_dynamic, IsInfLayerTest::getTestCaseName);
} // namespace
