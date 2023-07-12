// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/convert_color_i420.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "negative_layer_support_test.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<ov::Shape> inShapes_nhwc = {{1, 10, 10, 1}};

const std::vector<ov::element::Type> inTypes = {ov::element::u8, ov::element::f32};

const auto testCase_values = ::testing::Combine(::testing::ValuesIn(inShapes_nhwc),
                                                ::testing::ValuesIn(inTypes),
                                                ::testing::Bool(),
                                                ::testing::Bool(),
                                                ::testing::Values(ov::test::utils::DEVICE_GNA));

GNA_UNSUPPPORTED_LAYER_NEG_TEST(ConvertColorI420LayerTest, "The plugin does not support layer", testCase_values)

}  // namespace
