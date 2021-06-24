// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fake_quantize_and_max_pool_transformation.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<ngraph::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

const std::vector<ngraph::builder::subgraph::FakeQuantizeOnData> fakeQuantizes = {
    { 256ul, {}, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
};

// FakeQuantizeOnData

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FakeQuantizeAndMaxPoolTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::Values(ngraph::Shape({ 1, 32, 72, 48 })),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(fakeQuantizes)),
    FakeQuantizeAndMaxPoolTransformation::getTestCaseName);
}  // namespace
