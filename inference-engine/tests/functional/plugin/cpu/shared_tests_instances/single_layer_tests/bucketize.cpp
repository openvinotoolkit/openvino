// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/bucketize.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ov::test::subgraph;

const std::vector<ov::Shape> dataShapes = {
    {1, 20, 20},
    {2, 3, 50, 50}
};

const std::vector<ov::Shape> bucketsShapes = {
    {5},
    {20},
    {100}
};

const std::vector<ov::test::InputShape> dataShapesDynamic = {
    {{ngraph::Dimension(1, 10), ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()},
     {{1, 20, 20}, {3, 16, 16}, {10, 16, 16}}},
    {{ngraph::Dimension(1, 10), 3, 50, 50}, {{2, 3, 50, 50}, {2, 3, 50, 50}, {2, 3, 50, 50}}}};

const std::vector<ov::test::InputShape> bucketsShapesDynamic = {{{ngraph::Dimension::dynamic()}, {{5}, {20}, {100}}}};

const std::vector<ov::test::ElementType> inPrc = {
    ov::element::f32,
    ov::element::i64,
    ov::element::i32
};

const std::vector<ov::test::ElementType> netPrc = {
    ov::element::i64,
    ov::element::i32
};

const auto test_Bucketize_right_edge = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(dataShapes)),
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bucketsShapes)),
    ::testing::Values(true),
    ::testing::ValuesIn(inPrc),
    ::testing::ValuesIn(inPrc),
    ::testing::ValuesIn(netPrc),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto test_Bucketize_left_edge = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(dataShapes)),
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bucketsShapes)),
    ::testing::Values(false),
    ::testing::ValuesIn(inPrc),
    ::testing::ValuesIn(inPrc),
    ::testing::ValuesIn(netPrc),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);


const auto test_Bucketize_right_edge_Dynamic = ::testing::Combine(
    ::testing::ValuesIn(dataShapesDynamic),
    ::testing::ValuesIn(bucketsShapesDynamic),
    ::testing::Values(true),
    ::testing::ValuesIn(inPrc),
    ::testing::ValuesIn(inPrc),
    ::testing::ValuesIn(netPrc),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto test_Bucketize_left_edge_Dynamic = ::testing::Combine(
    ::testing::ValuesIn(dataShapesDynamic),
    ::testing::ValuesIn(bucketsShapesDynamic),
    ::testing::Values(false),
    ::testing::ValuesIn(inPrc),
    ::testing::ValuesIn(inPrc),
    ::testing::ValuesIn(netPrc),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsBucketize_right, BucketizeLayerTest, test_Bucketize_right_edge, BucketizeLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsBucketize_left, BucketizeLayerTest, test_Bucketize_left_edge, BucketizeLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsBucketize_right_Dynamic, BucketizeLayerTest, test_Bucketize_right_edge_Dynamic, BucketizeLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsBucketize_left_Dynamic, BucketizeLayerTest, test_Bucketize_left_edge_Dynamic, BucketizeLayerTest::getTestCaseName);
