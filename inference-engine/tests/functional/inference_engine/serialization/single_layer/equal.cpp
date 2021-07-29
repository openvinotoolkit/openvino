
// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "comparison_ops.hpp"

namespace {
TEST_P(ComparisonLayerTest, Serialize) {
    Serialize();
    }

const auto SerializeEqualTestParams = ::testing::Combine(
    ::testing::ValuesIn(CommonTestUtils::combineParams(inputShapes)),
    ::testing::ValuesIn(inputsPrecisions),
    ::testing::Values(ngraph::helpers::ComparisonTypes::EQUAL),
    ::testing::ValuesIn(secondInputTypes),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(CommonTestUtils::DEVICE_CPU),
    ::testing::Values(additional_config));


INSTANTIATE_TEST_SUITE_P(smoke_Equal, ComparisonLayerTest, SerializeEqualTestParams, ComparisonLayerTest::getTestCaseName);
} // namespace
