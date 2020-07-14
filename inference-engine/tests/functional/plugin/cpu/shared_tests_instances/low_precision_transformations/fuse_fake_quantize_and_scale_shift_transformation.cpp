// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fuse_fake_quantize_and_scale_shift_transformation.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ngraph_functions/low_precision_transformations/fuse_fake_quantize_and_scale_shift_function.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsFactory::createParamsU8I8()
};

const std::vector<ngraph::builder::subgraph::FakeQuantizeOnData> fakeQuantizeOnDataValues = {
    { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
    {
        256ul,
        { 1ul, 3ul, 1ul, 1ul },
        { 0.f, 0.f, 0.f },
        { 2.55f / 10.f, 2.55f / 5.f, 2.55f / 2.f },
        { 0.f, 0.f, 0.f },
        { 2.55f / 10.f, 2.55f / 5.f, 2.55f / 2.f }
    },
};

INSTANTIATE_TEST_CASE_P(LPT, FuseFakeQuantizeAndScaleShiftTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::SizeVector({ 1, 3, 9, 9 })),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(fakeQuantizeOnDataValues)),
    FuseFakeQuantizeAndScaleShiftTransformation::getTestCaseName);
}  // namespace
