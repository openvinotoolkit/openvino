// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/multiply_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    //ov::element::f16
};

const std::vector<LayerTestsDefinitions::MultiplyTestValues> params = {
    {false,
     {256ul, ov::Shape{1, 1, 1, 1}, {0.f}, {2.55f}, {0.f}, {2.55f}},
     false,
     {256ul, ov::Shape{1, 1, 1, 1}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f}},
     {256ul, ov::Shape{1, 1, 1, 1}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f}},
     ov::element::dynamic,  // ov::element::i8
     false},
    {false,
     {256ul, ov::Shape{1, 1, 1, 1}, {0.f}, {2.55f}, {0.f}, {2.55f}},
     false,
     {256ul, ov::Shape{1, 1, 1, 1}, {0.f}, {2.55f}, {0.f}, {2.55f}},
     {256ul, ov::Shape{1, 1, 1, 1}, {0.f}, {2.55f}, {0.f}, {2.55f}},
     ov::element::dynamic,  // ov::element::u8
     false},
    {true,
     {256ul, ov::Shape{1, 1, 1, 1}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f}},
     false,
     {256ul, ov::Shape{1, 1, 1, 1}, {0.f}, {2.55f}, {0.f}, {2.55f}},
     {256ul, ov::Shape{1, 1, 1, 1}, {0.f}, {2.55f}, {0.f}, {2.55f}},
     ov::element::dynamic,  // ov::element::u8
     false},
    {true,
     {256ul, ov::Shape{1, 1, 1, 1}, {0.f}, {2.55f}, {0.f}, {2.55f}},
     false,
     {256ul, ov::Shape{1, 1, 1, 1}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f}},
     {256ul, ov::Shape{1, 1, 1, 1}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f}},
     ov::element::dynamic,  // ov::element::i8
     false},
    {false,
     {256ul, ov::Shape{1, 1, 1, 1}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f}},
     true,
     {256ul, ov::Shape{1, 1, 1, 1}, {0.f}, {2.55f}, {0.f}, {2.55f}},
     {256ul, ov::Shape{1, 1, 1, 1}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f}},
     ov::element::dynamic,  // ov::element::i8
     false},
    {false,
     {256ul, ov::Shape{1, 1, 1, 1}, {-1.28f}, {1.27f}, {-128.f}, {1.27f}},
     false,
     {256ul, ov::Shape{1, 1, 1, 1}, {0.f}, {2.55f}, {0.f}, {2.55f}},
     {256ul, ov::Shape{1, 1, 1, 1}, {0.f}, {2.55f}, {0.f}, {2.55f}},
     ov::element::dynamic,  // ov::element::u8
     false},
    {false,
     {256ul, ov::Shape{1, 1, 1, 1}, {0.f}, {2.55f}, {0.f}, {2.55f}},
     true,
     {256ul, ov::Shape{1, 1, 1, 1}, {-1.27f}, {1.28f}, {-1.27f}, {1.28f}},
     {256ul, ov::Shape{1, 1, 1, 1}, {0.f}, {2.55f}, {0.f}, {2.55f}},
     ov::element::dynamic,  // ov::element::u8
     false},
    {false, {}, false, {}, {}, ov::element::dynamic /* ov::element::f32 */, false},
    {true, {}, true, {}, {}, ov::element::dynamic /* ov::element::f32 */, false},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, MultiplyTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::PartialShape({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(params)),
    MultiplyTransformation::getTestCaseName);
}  // namespace
