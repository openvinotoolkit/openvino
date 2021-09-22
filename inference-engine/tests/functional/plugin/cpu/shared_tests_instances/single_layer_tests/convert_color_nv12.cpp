// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/convert_color_nv12.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<ngraph::Shape> inShapes_nhwc = {
    {1, 224, 224, 1}
};

const std::vector<ngraph::element::Type> inTypes = {
        ngraph::element::u8, ngraph::element::f32
};

const auto testCase_values = ::testing::Combine(
    ::testing::ValuesIn(inShapes_nhwc),
    ::testing::ValuesIn(inTypes),
    ::testing::Values(true, false),
    ::testing::Values(true, false),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);


INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColotNV12, ConvertColorNV12LayerTest, testCase_values, ConvertColorNV12LayerTest::getTestCaseName);
