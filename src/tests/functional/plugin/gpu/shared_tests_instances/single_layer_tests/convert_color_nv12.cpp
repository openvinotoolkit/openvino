// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/convert_color_nv12.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<ov::Shape> inShapes_nhwc = {
    {1, 10, 10, 1}
};

const std::vector<ov::element::Type> inTypes = {
    ov::element::u8,
    ov::element::f32
};

const auto testCase_values = ::testing::Combine(
    ::testing::ValuesIn(inShapes_nhwc),
    ::testing::ValuesIn(inTypes),
    ::testing::Bool(),
    ::testing::Bool(),
    ::testing::Values(CommonTestUtils::DEVICE_GPU)
);


INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorNV12, ConvertColorNV12LayerTest, testCase_values, ConvertColorNV12LayerTest::getTestCaseName);

const auto testCase_accuracy_values = ::testing::Combine(
        ::testing::Values(ov::Shape{1, 16*6, 16, 1}),
        ::testing::Values(ov::element::u8),
        ::testing::Bool(),
        ::testing::Bool(),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorNV12_acc,
                         ConvertColorNV12AccuracyTest,
                         testCase_accuracy_values,
                         ConvertColorNV12LayerTest::getTestCaseName);

const auto testCase_accuracy_values_nightly = ::testing::Combine(
        ::testing::Values(ov::Shape{1, 256*256, 256, 1}),
        ::testing::Values(ov::element::u8),
        ::testing::Values(false),
        ::testing::Values(true),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(nightly_TestsConvertColorNV12_acc,
                         ConvertColorNV12AccuracyTest,
                         testCase_accuracy_values_nightly,
                         ConvertColorNV12LayerTest::getTestCaseName);

}  // namespace
