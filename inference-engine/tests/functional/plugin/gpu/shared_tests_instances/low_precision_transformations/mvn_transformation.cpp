// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/mvn_transformation.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
    const std::vector<element::Type> precisions = {
        element::f32,
        element::f16
    };

    const std::vector<ngraph::Shape> inputAndQuantizationShapes = {
        Shape{ 1ul, 4ul, 16ul, 16ul },
    };

    const std::vector<AxisSet> reductionAxes = { { 2, 3 }, { 1, 2, 3 } };

    const std::vector<bool> normalizeVariance = { true, false };

    INSTANTIATE_TEST_SUITE_P(smoke_LPT, MVNTransformation,
        ::testing::Combine(
            ::testing::ValuesIn(precisions),
            ::testing::ValuesIn(inputAndQuantizationShapes),
            ::testing::Values(CommonTestUtils::DEVICE_GPU),
            ::testing::ValuesIn(reductionAxes),
            ::testing::ValuesIn(normalizeVariance)),
        MVNTransformation::getTestCaseName);
}  // namespace
