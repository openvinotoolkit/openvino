// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fake_quantize_transformation.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ngraph_functions/low_precision_transformations/fake_quantize_function.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::pass::low_precision;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    // can not be passed to plugin
    // nGraph: I8 -> FP32 Convert is not supported
    // LayerTestsUtils::LayerTransformationParamsFactory::createParams(),
    // LayerTestsUtils::LayerTransformationParamsFactory::createParamsI8I8(),
    LayerTestsUtils::LayerTransformationParamsFactory::createParamsU8I8()
};

const std::vector<ngraph::builder::subgraph::FakeQuantizeOnData> fakeQuantizeOnDataValues = {
    { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
    { 256ul, { 1ul }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
    // nGraph: I8->FP32 Convert is not supported
    // { 256ul, {}, { -1.28f} , { 1.27f }, { -1.28f} , { 1.27f } },
    // { 256ul, { 1ul }, { -1.28f} , { 1.27f } }
};

// TODO: add something to avoid cleanup and enable
INSTANTIATE_TEST_CASE_P(smoke_LPT, FakeQuantizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::SizeVector({ 1, 32, 72, 48 })),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(fakeQuantizeOnDataValues)),
    FakeQuantizeTransformation::getTestCaseName);
}  // namespace
