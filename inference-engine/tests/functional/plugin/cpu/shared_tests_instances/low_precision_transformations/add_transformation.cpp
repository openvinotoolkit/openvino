// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/add_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<LayerTestsUtils::LayerTransformation::LptVersion> versions = {
    LayerTestsUtils::LayerTransformation::LptVersion::cnnNetwork,
    LayerTestsUtils::LayerTransformation::LptVersion::nGraph
};

const std::vector<LayerTestsDefinitions::AddTestValues> params = {
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -128.f }, { 127.f } },
        false,
        {InferenceEngine::Precision::I8}, {InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -128.f }, { 127.f }, { -128.f }, { 127.f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        false,
        {InferenceEngine::Precision::I8}, {InferenceEngine::Precision::FP32, InferenceEngine::Precision::I8}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -128.f }, { 127.f } },
        true,
        {InferenceEngine::Precision::I8}, {InferenceEngine::Precision::I8, InferenceEngine::Precision::FP32}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -128.f }, { 127.f }, { -128.f }, { 127.f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        true,
        {InferenceEngine::Precision::I8}, {InferenceEngine::Precision::I8, InferenceEngine::Precision::FP32}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -127.f }, { 128.f } },
        false,
        {InferenceEngine::Precision::U8}, {InferenceEngine::Precision::FP32, InferenceEngine::Precision::U8}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -128.f }, { 127.f }, { -128.f }, { 127.f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        false,
        {InferenceEngine::Precision::U8}, {InferenceEngine::Precision::FP32, InferenceEngine::Precision::U8}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { -127.f }, { 128.f } },
        true,
        {InferenceEngine::Precision::U8}, {InferenceEngine::Precision::U8, InferenceEngine::Precision::FP32}
    },
    {
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { -128.f }, { 127.f }, { -128.f }, { 127.f } },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 255.f }, { 0.f }, { 255.f } },
        true,
        {InferenceEngine::Precision::U8}, {InferenceEngine::Precision::U8, InferenceEngine::Precision::FP32}
    },
    { {}, {}, false }, { {}, {}, true },
};

INSTANTIATE_TEST_CASE_P(LPT, AddTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::SizeVector({ 1, 3, 16, 16 })),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(versions),
        ::testing::ValuesIn(params)),
    AddTransformation::getTestCaseName);
}  // namespace
