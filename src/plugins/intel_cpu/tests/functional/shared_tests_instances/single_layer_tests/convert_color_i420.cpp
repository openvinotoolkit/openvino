// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/convert_color_i420.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<ov::Shape> inShapes_nhwc = {
    {1, 10, 10, 1}
};

const std::vector<ov::element::Type> inTypes = {
        ov::element::u8, ov::element::f32
};

const auto testCase_values = ::testing::Combine(
    ::testing::ValuesIn(inShapes_nhwc),
    ::testing::ValuesIn(inTypes),
    ::testing::Bool(),
    ::testing::Bool(),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);


INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorI420, ConvertColorI420LayerTest, testCase_values, ConvertColorI420LayerTest::getTestCaseName);

const auto testCase_accuracy_values = ::testing::Combine(
        ::testing::Values(ov::Shape{1, 16*6, 16, 1}),
        ::testing::Values(ov::element::u8),
        ::testing::Values(false),
        ::testing::Values(true),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorI420_acc,
                         ConvertColorI420AccuracyTest,
                         testCase_accuracy_values,
                         ConvertColorI420LayerTest::getTestCaseName);

const auto testCase_accuracy_values_nightly = ::testing::Combine(
        ::testing::Values(ov::Shape{1, 256*256, 256, 1}),
        ::testing::Values(ov::element::u8),
        ::testing::Values(false),
        ::testing::Values(true),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(nightly_TestsConvertColorI420_acc,
                         ConvertColorI420AccuracyTest,
                         testCase_accuracy_values_nightly,
                         ConvertColorI420LayerTest::getTestCaseName);

}  // namespace
