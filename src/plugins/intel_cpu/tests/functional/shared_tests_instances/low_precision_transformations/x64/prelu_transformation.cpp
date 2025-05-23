// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/prelu_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> precisions = {
        ov::element::f32
};

std::vector<PReluTestValues> testValues = {
    { {}, false},
    { { 256ul, ov::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} }, false },
    { { 256ul, ov::Shape({}), {-12.8f}, {12.7f}, {-12.8f}, {12.7f} }, true },
    { { 256ul, ov::Shape({}), {12.75f}, {25.5f}, {12.75f}, {25.5f} }, true },
    { { 256ul, ov::Shape({}), {-12.8f / 2.f}, {12.7f}, {-12.8f / 2.f}, {12.7f} }, true }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, PReluTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::Values(ov::PartialShape({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(testValues)),
    PReluTransformation::getTestCaseName);
}  // namespace
