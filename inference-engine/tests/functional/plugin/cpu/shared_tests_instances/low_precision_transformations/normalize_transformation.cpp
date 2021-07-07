// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/normalize_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    //ngraph::element::f16
};

const std::vector<std::pair<ngraph::Shape, ngraph::Shape> > inputAndQuantizationShapes = {
    { ngraph::Shape({ 1ul, 4ul, 16ul, 16ul }), ngraph::Shape({ 1ul }) },
    { ngraph::Shape({ 1ul, 4ul, 16ul, 16ul }), ngraph::Shape({ 1ul, 4ul, 1ul, 1ul }) },
};

const std::vector<std::vector<uint64_t>> axes = {
    { 1 }, { 1, 2, 3 }
};

const std::vector<bool> fuseMultiplyValues = { true, false };

const std::vector<bool> shiftValues = { true, false };

INSTANTIATE_TEST_SUITE_P(smoke_LPT, NormalizeL2Transformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(inputAndQuantizationShapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(axes),
        ::testing::ValuesIn(fuseMultiplyValues),
        ::testing::ValuesIn(shiftValues)),
    NormalizeL2Transformation::getTestCaseName);
}  // namespace
