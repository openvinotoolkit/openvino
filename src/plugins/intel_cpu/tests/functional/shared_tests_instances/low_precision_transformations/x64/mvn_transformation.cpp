// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/mvn_transformation.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> precisions = {
        ov::element::f32
};

const std::vector<ov::PartialShape> inputAndQuantizationShapes = {
    { 1ul, 4ul, 16ul, 16ul },
};

const std::vector<ov::AxisSet> reductionAxes = {{2, 3}, {1, 2, 3}};

const std::vector<bool> normalizeVariance = { true, false };

INSTANTIATE_TEST_SUITE_P(smoke_LPT, MVNTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(inputAndQuantizationShapes),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(reductionAxes),
        ::testing::ValuesIn(normalizeVariance)),
    MVNTransformation::getTestCaseName);
}  // namespace
