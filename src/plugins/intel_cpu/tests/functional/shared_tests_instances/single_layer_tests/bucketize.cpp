// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/bucketize.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::BucketizeLayerTest;

namespace {
const std::vector<std::vector<ov::Shape>> input_shapes_static = {
        //data_shape, bucket_shape
        {{ 1, 20, 20 }, {5}},
        {{ 1, 20, 20 }, {20}},
        {{ 1, 20, 20 }, {100}},
        {{ 2, 3, 50, 50 }, {5}},
        {{ 2, 3, 50, 50 }, {20}},
        {{ 2, 3, 50, 50 }, {100}}
};

const std::vector<ov::element::Type> in_types = {
    ov::element::f32,
    ov::element::f16,
    ov::element::i64,
    ov::element::i32
};

const std::vector<ov::element::Type> model_types = {
    ov::element::i64,
    ov::element::i32
};

const auto test_Bucketize_right_edge = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
    ::testing::Values(true),
    ::testing::ValuesIn(in_types),
    ::testing::ValuesIn(in_types),
    ::testing::ValuesIn(model_types),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto test_Bucketize_left_edge = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
    ::testing::Values(false),
    ::testing::ValuesIn(in_types),
    ::testing::ValuesIn(in_types),
    ::testing::ValuesIn(model_types),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsBucketize_right, BucketizeLayerTest, test_Bucketize_right_edge, BucketizeLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsBucketize_left, BucketizeLayerTest, test_Bucketize_left_edge, BucketizeLayerTest::getTestCaseName);
} //  namespace
