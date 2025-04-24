// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/interpolate_transformation.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> precisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<std::pair<ov::PartialShape, ov::Shape>> shapes = {
    {{1, 4, 16, 16}, {32, 32}},
    {{1, 2, 48, 80}, {50, 60}},
};

const std::vector<interpAttributes> interpAttrs = {
        interpAttributes(
            ov::AxisSet{2, 3},
            "nearest",
            false,
            false,
            {0},
            {0}),
        interpAttributes(
            ov::AxisSet{2, 3},
            "nearest",
            false,
            true,
            {0},
            {0}),
        interpAttributes(
            ov::AxisSet{2, 3},
            "linear",
            false,
            false,
            {0},
            {0}),
};

const auto combineValues = ::testing::Combine(
    ::testing::ValuesIn(precisions),
    ::testing::ValuesIn(shapes),
    ::testing::Values(ov::test::utils::DEVICE_GPU),
    ::testing::ValuesIn(interpAttrs));

INSTANTIATE_TEST_SUITE_P(smoke_LPT, InterpolateTransformation, combineValues, InterpolateTransformation::getTestCaseName);
}  // namespace
