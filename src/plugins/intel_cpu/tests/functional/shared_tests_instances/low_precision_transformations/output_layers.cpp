// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/output_layers.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ov::pass::low_precision;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsFactory::createParams(),
    // LayerTestsUtils::LayerTransformationParamsFactory::createParamsI8I8(),
    // LayerTestsUtils::LayerTransformationParamsFactory::createParamsU8I8()
};


INSTANTIATE_TEST_SUITE_P(smoke_LPT, OutputLayers,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::SizeVector({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues)),
    OutputLayers::getTestCaseName);
}  // namespace
