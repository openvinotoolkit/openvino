// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/multiply_with_one_parent_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32
};

const std::vector<MultiplyWithOneParentTransformationValues> values = {
    {
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, MultiplyWithOneParentTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::PartialShape({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(values)),
    MultiplyWithOneParentTransformation::getTestCaseName);
}  // namespace
