// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/bucketize.hpp"

#include <vector>

using ov::test::BucketizeLayerTest;

namespace {

const std::vector<std::vector<ov::Shape>> input_shapes_static  = {
    {{40, 22, 13, 9}, {5}},
    {{6, 7, 3, 2, 8}, {5}},
    {{6, 7, 3, 2, 8, 5}, {5}},
    {{40, 22, 13, 9}, {100}},
    {{6, 7, 3, 2, 8}, {100}},
    {{6, 7, 3, 2, 8, 5}, {100}},
};

const std::vector<bool> with_right_bound = {true, false};

const std::vector<ov::element::Type> out_precision = {
    ov::element::i32,
    ov::element::i64
};

const std::vector<ov::element::Type> in_buckets_precision = {
    ov::element::f16,
    ov::element::f32,
    ov::element::i32,
    ov::element::i64,
    ov::element::i8,
    ov::element::u8
};

// We won't test FP32 and FP16 together as it won't make sense for now
// as ngraph reference implementation use FP32 for FP16 case

INSTANTIATE_TEST_SUITE_P(smoke_Bucketize_input_fp16,
                         BucketizeLayerTest,
                         testing::Combine(testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                          testing::ValuesIn(with_right_bound),
                                          testing::Values(ov::element::f16),
                                          testing::Values(ov::element::f16,
                                                          ov::element::i32,
                                                          ov::element::i64,
                                                          ov::element::i8,
                                                          ov::element::u8),
                                          testing::ValuesIn(out_precision),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         BucketizeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Bucketize_input_fp32,
                         BucketizeLayerTest,
                         testing::Combine(testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                          testing::ValuesIn(with_right_bound),
                                          testing::Values(ov::element::f32),
                                          testing::Values(ov::element::f32,
                                                          ov::element::i32,
                                                          ov::element::i64,
                                                          ov::element::i8,
                                                          ov::element::u8),
                                          testing::ValuesIn(out_precision),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         BucketizeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Bucketize_input_i32,
                         BucketizeLayerTest,
                         testing::Combine(testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                          testing::ValuesIn(with_right_bound),
                                          testing::Values(ov::element::i32),
                                          testing::ValuesIn(in_buckets_precision),
                                          testing::ValuesIn(out_precision),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         BucketizeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Bucketize_input_i64,
                         BucketizeLayerTest,
                         testing::Combine(testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                          testing::ValuesIn(with_right_bound),
                                          testing::Values(ov::element::i64),
                                          testing::ValuesIn(in_buckets_precision),
                                          testing::ValuesIn(out_precision),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         BucketizeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Bucketize_input_i8,
                         BucketizeLayerTest,
                         testing::Combine(testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                          testing::ValuesIn(with_right_bound),
                                          testing::Values(ov::element::i8),
                                          testing::ValuesIn(in_buckets_precision),
                                          testing::ValuesIn(out_precision),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         BucketizeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Bucketize_input_u8,
                         BucketizeLayerTest,
                         testing::Combine(testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                          testing::ValuesIn(with_right_bound),
                                          testing::Values(ov::element::u8),
                                          testing::ValuesIn(in_buckets_precision),
                                          testing::ValuesIn(out_precision),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         BucketizeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Bucketize_empty_boundary_f16,
                         BucketizeLayerTest,
                         testing::Combine(testing::Values(ov::test::static_shapes_to_test_representation({{6, 7, 3, 2, 8, 5}, {0}})),
                                          testing::ValuesIn(with_right_bound),
                                          testing::Values(ov::element::f16),
                                          testing::Values(ov::element::f16),
                                          testing::ValuesIn(out_precision),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         BucketizeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Bucketize_empty_boundary_f32,
                         BucketizeLayerTest,
                         testing::Combine(testing::Values(ov::test::static_shapes_to_test_representation({{6, 7, 3, 2, 8, 5}, {0}})),
                                          testing::ValuesIn(with_right_bound),
                                          testing::Values(ov::element::f32),
                                          testing::Values(ov::element::f32),
                                          testing::ValuesIn(out_precision),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         BucketizeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Bucketize_empty_boundary_i8,
                         BucketizeLayerTest,
                         testing::Combine(testing::Values(ov::test::static_shapes_to_test_representation({{6, 7, 3, 2, 8, 5}, {0}})),
                                          testing::ValuesIn(with_right_bound),
                                          testing::Values(ov::element::i8),
                                          testing::Values(ov::element::i8),
                                          testing::ValuesIn(out_precision),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         BucketizeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Bucketize_empty_boundary_u8,
                         BucketizeLayerTest,
                         testing::Combine(testing::Values(ov::test::static_shapes_to_test_representation({{6, 7, 3, 2, 8, 5}, {0}})),
                                          testing::ValuesIn(with_right_bound),
                                          testing::Values(ov::element::u8),
                                          testing::Values(ov::element::u8),
                                          testing::ValuesIn(out_precision),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         BucketizeLayerTest::getTestCaseName);

}  // namespace
