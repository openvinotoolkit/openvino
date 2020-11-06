// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/unsqueeze_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::pass::low_precision;

namespace {
    const std::vector<InferenceEngine::Precision> precisions = {
        InferenceEngine::Precision::FP32,
    };


    const std::vector<LayerTransformation::Params> trasformationParamValues = {
        LayerTestsUtils::LayerTransformationParamsFactory::createParamsU8I8(),
        LayerTestsUtils::LayerTransformationParamsFactory::createParamsI8I8().setUpdatePrecisions(false),
        LayerTestsUtils::LayerTransformationParamsFactory::createParamsI8I8().setUpdatePrecisions(true),
    };

    const std::vector<LayerTestsDefinitions::UnsqueezeTransformationParam> params = {
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -12.8f }, { 12.7f }, { -12.8f }, { 12.7f } },
            { 0.0, 3.0 },
            { 3, 3, 5}
        },
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -12.8f }, { 12.7f }, { -12.8f }, { 12.7f } },
            { 0.0, 1.0 },
            { 3, 3, 3 }
        },
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -12.8f }, { 12.7f }, { -12.8f }, { 12.7f } },
            { 3.0 },
            { 3, 4, 5, 6 }
        },
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -12.8f }, { 12.7f }, { -12.8f }, { 12.7f } },
            { 0.0, 3.0 },
            { 1, 32, 2}
        },
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -12.8f }, { 12.7f }, { -12.8f }, { 12.7f } },
            { 0.0, 1.0 },
            { 46, 128, 2 }
        }
    };

    INSTANTIATE_TEST_CASE_P(LPT, UnsqueezeTransformation,
        ::testing::Combine(
            ::testing::ValuesIn(precisions),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::ValuesIn(trasformationParamValues),
            ::testing::ValuesIn(params)),
        UnsqueezeTransformation::getTestCaseName);
}  // namespace
