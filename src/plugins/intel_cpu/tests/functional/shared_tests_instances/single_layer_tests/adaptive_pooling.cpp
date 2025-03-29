// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/adaptive_pooling.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::AdaPoolLayerTest;

const std::vector<ov::element::Type> types = {
        ov::element::f16,
        ov::element::f32
};

const std::vector<std::vector<ov::Shape>> input_shapes_3d_static = {
        {{ 1, 2, 1}},
        {{ 1, 1, 3 }},
        {{ 3, 17, 5 }}
};

const auto ada_pool_3d_cases = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_3d_static)),
        ::testing::ValuesIn(std::vector<std::vector<int>>{ {1}, {3}, {5} }),
        ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),
        ::testing::ValuesIn(types),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsAdaPool3D, AdaPoolLayerTest, ada_pool_3d_cases, AdaPoolLayerTest::getTestCaseName);

const std::vector<std::vector<ov::Shape>> input_shapes_4d_static = {
        {{ 1, 2, 1, 2 }},
        {{ 1, 1, 3, 2 }},
        {{ 3, 17, 5, 1 }}
};

const auto ada_pool_4d_cases = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_4d_static)),
        ::testing::ValuesIn(std::vector<std::vector<int>>{ {1, 1}, {3, 5}, {5, 5} }),
        ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),
        ::testing::ValuesIn(types),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsAdaPool4D, AdaPoolLayerTest, ada_pool_4d_cases, AdaPoolLayerTest::getTestCaseName);

const std::vector<std::vector<ov::Shape>> input_shapes_5d_static = {
        {{ 1, 2, 1, 2, 2 }},
        {{ 1, 1, 3, 2, 3 }},
        {{ 3, 17, 5, 1, 2 }}
};

const auto ada_pool_5d_cases = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_5d_static)),
        ::testing::ValuesIn(std::vector<std::vector<int>>{ {1, 1, 1}, {3, 5, 3}, {5, 5, 5} }),
        ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),
        ::testing::ValuesIn(types),
        ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsAdaPool5D, AdaPoolLayerTest, ada_pool_5d_cases, AdaPoolLayerTest::getTestCaseName);
} //  namespace
