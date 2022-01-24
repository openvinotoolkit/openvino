// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/convert_color_nv12.hpp"

using namespace LayerTestsDefinitions;

namespace {

TEST_P(ConvertColorNV12LayerTest, Serialize) {
        Serialize();
    }

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

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs, ConvertColorNV12LayerTest, testCase_values, ConvertColorNV12LayerTest::getTestCaseName);

} // namespace
