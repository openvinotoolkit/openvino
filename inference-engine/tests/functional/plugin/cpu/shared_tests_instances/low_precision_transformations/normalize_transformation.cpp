// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/normalize_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<InferenceEngine::Precision> precisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::pair<ngraph::Shape, ngraph::Shape> > inputAndQuantizationShapes = {
    { ngraph::Shape({ 1ul, 4ul, 16ul, 16ul }), ngraph::Shape({ 1ul }) },
    // TODO: different scales initialization is not implemented yet
    { ngraph::Shape({ 1ul, 4ul, 16ul, 16ul }), ngraph::Shape({ 1ul, 4ul, 1ul, 1ul }) },
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsFactory::createParams(),
    LayerTestsUtils::LayerTransformationParamsFactory::createParamsI8I8(),
    LayerTestsUtils::LayerTransformationParamsFactory::createParamsU8I8()
};

const std::vector<LayerTestsUtils::LayerTransformation::LptVersion> versionValues = {
    LayerTestsUtils::LayerTransformation::LptVersion::cnnNetwork,
    // NormalizeIE expected instead NormalizeL2
    // LayerTestsUtils::LayerTransformation::LptVersion::nGraph
};

const std::vector<bool> fuseMultiplyValues = { /* true, */ false };

const std::vector<bool> shiftValues = { /* true, */ false };

INSTANTIATE_TEST_CASE_P(LPT, NormalizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(inputAndQuantizationShapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(versionValues),
        ::testing::ValuesIn(fuseMultiplyValues),
        ::testing::ValuesIn(shiftValues)),
    NormalizeTransformation::getTestCaseName);
}  // namespace
