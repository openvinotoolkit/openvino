// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/normalize_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

const std::vector<std::pair<ngraph::Shape, ngraph::Shape> > inputAndQuantizationShapes = {
    { ngraph::Shape({ 1ul, 4ul, 16ul, 16ul }), ngraph::Shape({ 1ul }) },
    //{ ngraph::Shape({ 1ul, 4ul, 16ul, 16ul }), ngraph::Shape({ 1ul, 4ul, 1ul, 1ul }) },
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsFactory::createParams()
};

const std::vector<LayerTestsUtils::LayerTransformation::LptVersion> versionValues = {
    LayerTestsUtils::LayerTransformation::LptVersion::cnnNetwork,
    // LayerTestsUtils::LayerTransformation::LptVersion::nGraph
};

const std::vector<bool> fuseMultiplyValues = { true, false };

const std::vector<bool> shiftValues = { true, false };

INSTANTIATE_TEST_CASE_P(LPT, NormalizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(inputAndQuantizationShapes),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(versionValues),
        ::testing::ValuesIn(fuseMultiplyValues),
        ::testing::ValuesIn(shiftValues)),
    NormalizeTransformation::getTestCaseName);
}  // namespace
