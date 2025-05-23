// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/fake_convert.hpp"

namespace {
using ov::test::FakeConvertLayerTest;

const std::vector<std::vector<ov::Shape>> shapes = {{{2, 3, 4, 5}}};

const std::vector<ov::element::Type> data_precisions = {ov::element::f32, ov::element::f16, ov::element::bf16};

const std::vector<ov::element::Type> destination_precisions = {ov::element::f8e4m3, ov::element::f8e5m2};

const std::vector<bool> default_shift = {true, false};

const auto simple_fake_convert_params =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes)),
                       ::testing::Values(ov::Shape{1}),
                       ::testing::Values(ov::Shape{1}),
                       ::testing::ValuesIn(data_precisions),
                       ::testing::ValuesIn(destination_precisions),
                       ::testing::ValuesIn(default_shift),
                       ::testing::Values(ov::test::utils::DEVICE_CPU));

const auto broadcast_fake_convert_params =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes)),
                       ::testing::Values(ov::Shape{2, 3, 1, 1}),
                       ::testing::Values(ov::Shape{2, 3, 1, 1}),
                       ::testing::ValuesIn(data_precisions),
                       ::testing::ValuesIn(destination_precisions),
                       ::testing::ValuesIn(default_shift),
                       ::testing::Values(ov::test::utils::DEVICE_CPU));

const auto elementwise_fake_convert_params =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes)),
                       ::testing::Values(ov::Shape{2, 3, 4, 5}),
                       ::testing::Values(ov::Shape{2, 3, 4, 5}),
                       ::testing::ValuesIn(data_precisions),
                       ::testing::ValuesIn(destination_precisions),
                       ::testing::ValuesIn(default_shift),
                       ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_FakeConvert_simple,
                         FakeConvertLayerTest,
                         simple_fake_convert_params,
                         FakeConvertLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FakeConvert_broadcast,
                         FakeConvertLayerTest,
                         broadcast_fake_convert_params,
                         FakeConvertLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FakeConvert_elementwise,
                         FakeConvertLayerTest,
                         elementwise_fake_convert_params,
                         FakeConvertLayerTest::getTestCaseName);
}  // namespace
