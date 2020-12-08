// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/relu_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

std::vector<ngraph::builder::subgraph::FakeQuantizeOnData> testValues = {
    {},
    { 256ul, ngraph::Shape({}), {0.f}, {25.5f}, {0.f}, {25.5f} },
    { 256ul, ngraph::Shape({}), {-12.8f}, {12.7f}, {-12.8f}, {12.7f} },
    { 256ul, ngraph::Shape({}), {12.75f}, {25.5f}, {12.75f}, {25.5f} },
    { 256ul, ngraph::Shape({}), {-12.8f / 2.f}, {12.7f}, {-12.8f / 2.f}, {12.7f} }
};

INSTANTIATE_TEST_CASE_P(smoke_LPT, ReluTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::Values(InferenceEngine::SizeVector({ 1, 3, 16, 16 })),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::ValuesIn(testValues)),
    ReluTransformation::getTestCaseName);
}  // namespace
