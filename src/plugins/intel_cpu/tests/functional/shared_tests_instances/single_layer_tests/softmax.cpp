// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/softmax.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ov::test::subgraph;

namespace {

const std::vector<ov::test::ElementType> netPrecisions = {
    ov::element::Type_t::f32,
};

const std::vector<ov::Shape> inputStaticShape2D = {
        {1, 100},
        {100, 1},
        {10, 10},
};

const std::vector<ov::test::InputShape> inputDynamicShape2D = {
        {{ov::Dimension::dynamic(), 10}, {{1, 10}, {2, 10}, {10, 10}}},
        {{ov::Dimension(1, 10), 10}, {{1, 10}, {2, 10}, {10, 10}}},
        {{10, ov::Dimension::dynamic()}, {{10, 1}, {10, 5}, {10, 10}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 10}, {2, 10}, {10, 10}}}
};

const std::vector<int64_t> axis2D = {
    0, 1, -1
};

const auto params2D_static =
    testing::Combine(testing::ValuesIn(netPrecisions),
                     ::testing::Values(ov::element::dynamic),
                     ::testing::Values(ov::element::dynamic),
                     testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputStaticShape2D)),
                     testing::ValuesIn(axis2D),
                     testing::Values(ov::test::utils::DEVICE_CPU),
                     testing::Values(ov::AnyMap()));

const auto params2D_dynamic = testing::Combine(testing::ValuesIn(netPrecisions),
                                               ::testing::Values(ov::element::dynamic),
                                               ::testing::Values(ov::element::dynamic),
                                               testing::ValuesIn(inputDynamicShape2D),
                                               testing::ValuesIn(axis2D),
                                               testing::Values(ov::test::utils::DEVICE_CPU),
                                               testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_SUITE_P(
        smoke_SoftMax2D_static,
        SoftMax8LayerTest,
        params2D_static,
        SoftMax8LayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_SoftMax2D_dynamic,
        SoftMax8LayerTest,
        params2D_dynamic,
        SoftMax8LayerTest::getTestCaseName
);

const std::vector<ov::Shape> inputStaticShape4D = {
        {1, 100, 1, 1},
        {50, 100, 4, 1},
        {2, 100, 10, 1},
};

const std::vector<ov::test::InputShape> inputDynamicShape4D = {
        {{ov::Dimension::dynamic(), 100, ov::Dimension(1, 10), 1}, {{1, 100, 1, 1}, {100, 100, 5, 1}}},
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
         {{1, 100, 1, 1}, {50, 100, 4, 1}, {2, 100, 10, 1}}},
};

const std::vector<int64_t> axis4D = {0, 1, 2, 3, -1, -2, -3};

const auto params4Dstatic =
    testing::Combine(testing::ValuesIn(netPrecisions),
                     ::testing::Values(ov::element::dynamic),
                     ::testing::Values(ov::element::dynamic),
                     testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputStaticShape4D)),
                     testing::ValuesIn(axis4D),
                     testing::Values(ov::test::utils::DEVICE_CPU),
                     testing::Values(ov::AnyMap()));

const auto params4Ddynamic = testing::Combine(testing::ValuesIn(netPrecisions),
                                              ::testing::Values(ov::element::dynamic),
                                              ::testing::Values(ov::element::dynamic),
                                              testing::ValuesIn(inputDynamicShape4D),
                                              testing::ValuesIn(axis4D),
                                              testing::Values(ov::test::utils::DEVICE_CPU),
                                              testing::Values(ov::AnyMap()));

INSTANTIATE_TEST_SUITE_P(
        smoke_SoftMax4D_static,
        SoftMax8LayerTest,
        params2D_static,
        SoftMax8LayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_SoftMax4D_dynamic,
        SoftMax8LayerTest,
        params2D_dynamic,
        SoftMax8LayerTest::getTestCaseName
);

}  // namespace
