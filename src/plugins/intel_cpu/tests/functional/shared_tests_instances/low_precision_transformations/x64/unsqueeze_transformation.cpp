// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/unsqueeze_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ov::pass::low_precision;

namespace {
    const std::vector<ov::element::Type> precisions = {
        ov::element::f32
    };

    const std::vector<LayerTransformation::Params> trasformationParamValues = {
        LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8(),
        // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsI8I8().setUpdatePrecisions(false),
        // LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsI8I8().setUpdatePrecisions(true),
    };

    const std::vector<LayerTestsDefinitions::UnsqueezeTransformationParam> params = {
        {
            { 256ul, ov::Shape { 1, 1, 1 }, { -12.8f }, { 12.7f }, { -12.8f }, { 12.7f } },
            { 3.0 },
            { 3, 3, 5}
        },
        {
            { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -12.8f }, { 12.7f } },
            { 3.0 },
            { 3, 3, 3, 5 }
        },
        {
            { 256ul, ov::Shape { 1, 1, 1, 1 }, { -12.8f }, { 12.7f }, { -12.8f }, { 12.7f } },
            { 3.0 },
            { 3, 4, 5, 6 }
        },
        {
            { 256ul, ov::Shape { 1, 1 }, { -12.8f }, { 12.7f }, { -12.8f }, { 12.7f } },
            { 2.0, 3.0 },
            { 3, 4 }
        },
        {
            { 256ul, ov::Shape { 1, 1, 1, 1 }, { -12.8f }, { 12.7f }, { -12.8f }, { 12.7f } },
            { 4.0 },
            { 46, 128, 2, 3 }
        }
    };

    INSTANTIATE_TEST_SUITE_P(smoke_LPT, UnsqueezeTransformation,
        ::testing::Combine(
            ::testing::ValuesIn(precisions),
            ::testing::Values(ov::test::utils::DEVICE_CPU),
            ::testing::ValuesIn(trasformationParamValues),
            ::testing::ValuesIn(params)),
        UnsqueezeTransformation::getTestCaseName);
}  // namespace
