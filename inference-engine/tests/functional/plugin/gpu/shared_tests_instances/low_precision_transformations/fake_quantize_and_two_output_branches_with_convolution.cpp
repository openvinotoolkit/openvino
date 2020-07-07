// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fake_quantize_and_two_output_branches_with_convolution.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsFactory::createParamsU8I8()
};

const std::vector<LayerTestsUtils::LayerTransformation::LptVersion> versionValues = {
    LayerTestsUtils::LayerTransformation::LptVersion::cnnNetwork,
    LayerTestsUtils::LayerTransformation::LptVersion::nGraph
};

const std::vector<ngraph::builder::subgraph::FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::ActualValues> testValues = {
    {
        { 256ul, {}, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
        { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
        { 255ul, {1, 1, 1, 1}, { 0.f }, { 254.f }, { -127.f }, { 127.f } },
    }
};

// TODO: add something to avoid cleanup and enable
INSTANTIATE_TEST_CASE_P(LPT, FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::SizeVector({ 1, 32, 72, 48 })),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(versionValues),
        ::testing::ValuesIn(testValues)),
    FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation::getTestCaseName);
}  // namespace
