// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/power.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::PowerLayerTest;

std::vector<std::vector<ov::Shape>> input_shape_static = {
        {{1, 8}},
        {{2, 16}},
        {{3, 32}},
        {{4, 64}},
        {{5, 128}},
        {{6, 256}},
        {{7, 512}},
        {{8, 1024}}
};

std::vector<std::vector<float>> powers = {
        {0.0f},
        {0.5f},
        {1.0f},
        {1.1f},
        {1.5f},
        {2.0f},
};

std::vector<ov::element::Type> model_types = {
    ov::element::f32,
    ov::element::f16,
};

INSTANTIATE_TEST_SUITE_P(smoke_power, PowerLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shape_static)),
                                ::testing::ValuesIn(model_types),
                                ::testing::ValuesIn(powers),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        PowerLayerTest::getTestCaseName);
}  // namespace
