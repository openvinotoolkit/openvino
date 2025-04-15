// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fake_quantize_and_avg_pool_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> precisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<ov::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8()
};

const std::vector<ov::builder::subgraph::FakeQuantizeOnData> fakeQuantizes = {
    { 256ul, {}, { 0.f }, { 25.5f }, { 0.f }, { 25.5f } },
};

// FakeQuantizeOnData

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FakeQuantizeAndAvgPoolTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::Values(ov::PartialShape({ 1, 32, 72, 48 })),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(fakeQuantizes)),
    FakeQuantizeAndAvgPoolTransformation::getTestCaseName);
}  // namespace
