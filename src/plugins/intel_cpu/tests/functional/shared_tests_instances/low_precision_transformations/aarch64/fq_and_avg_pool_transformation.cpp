// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "low_precision_transformations/fake_quantize_and_avg_pool_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> precisions = {
    ov::element::f32
};

const ov::builder::subgraph::FakeQuantizeOnData fakeQuantizeI8 = {
    256ul, {}, {-12.8f}, {12.7f}, {-12.8f}, {12.7f}
};

const ov::builder::subgraph::FakeQuantizeOnData fakeQuantizeU8 = {
    256ul, {}, {0.f}, {25.5f}, {0.f}, {25.5f}
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT_i8,
                         FakeQuantizeAndAvgPoolTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::Values(ov::PartialShape({1, 32, 72, 48})),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(fakeQuantizeI8)),
                         FakeQuantizeAndAvgPoolTransformation::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LPT_u8,
                         FakeQuantizeAndAvgPoolTransformation,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::Values(ov::PartialShape({1, 32, 72, 48})),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(fakeQuantizeU8)),
                         FakeQuantizeAndAvgPoolTransformation::getTestCaseName);
}  // namespace
