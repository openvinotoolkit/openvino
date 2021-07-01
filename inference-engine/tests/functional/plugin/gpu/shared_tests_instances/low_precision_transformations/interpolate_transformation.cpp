// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/interpolate_transformation.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<std::pair<ngraph::Shape, ngraph::Shape>> shapes = {
    {{1, 4, 16, 16}, {32, 32}},
    {{1, 2, 48, 80}, {50, 60}},
};

const std::vector<interpAttributes> interpAttrs = {
        interpAttributes(
            ngraph::AxisSet{2, 3},
            "nearest",
            false,
            false,
            {0},
            {0}),
        interpAttributes(
            ngraph::AxisSet{2, 3},
            "nearest",
            false,
            true,
            {0},
            {0}),
        interpAttributes(
            ngraph::AxisSet{2, 3},
            "linear",
            false,
            false,
            {0},
            {0}),
};

const auto combineValues = ::testing::Combine(
    ::testing::ValuesIn(precisions),
    ::testing::ValuesIn(shapes),
    ::testing::Values(CommonTestUtils::DEVICE_GPU),
    ::testing::ValuesIn(interpAttrs));

INSTANTIATE_TEST_SUITE_P(smoke_LPT, InterpolateTransformation, combineValues, InterpolateTransformation::getTestCaseName);
}  // namespace
