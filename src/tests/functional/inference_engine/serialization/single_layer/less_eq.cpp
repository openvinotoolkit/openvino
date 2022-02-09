// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "comparison_ops.hpp"

using namespace LayerTestsDefinitions;
using namespace LayerTestsDefinitions::ComparisonParams;

namespace {
TEST_P(ComparisonLayerTest, Serialize) {
        Serialize();
    }

ComparisionOpsData data = {
    // inputsShape
    {
        {{1}, {{1}, {17}, {1, 1}, {2, 18}, {1, 1, 2}, {2, 2, 3}, {1, 1, 2, 3}}},
        {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
        {{2, 200}, {{1}, {200}, {1, 200}, {2, 200}, {2, 2, 200}}},
        {{1, 3, 20}, {{20}, {2, 1, 1}}},
        {{2, 17, 3, 4}, {{4}, {1, 3, 4}, {2, 1, 3, 4}}},
        {{2, 17, 3, 4}, {{4}, {1, 3, 4}, {141, 1, 3, 4}}},
        {{2, 1, 1, 3, 1}, {{1}, {1, 3, 4}, {2, 1, 3, 4}, {1, 1, 1, 1, 1}}},
    },
    // inputsPrecisions
    {
        InferenceEngine::Precision::FP64,
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::U32,
        InferenceEngine::Precision::BOOL,
    },
    // secondIinputsType
    {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
    },
    // additionalConfig
    {},
    // opType
    ngraph::helpers::ComparisonTypes::LESS_EQUAL,
    // ieInputPrecision
    InferenceEngine::Precision::UNSPECIFIED,
    // ieOutputPrecision
    InferenceEngine::Precision::UNSPECIFIED,
    // deviceName
    CommonTestUtils::DEVICE_CPU,
};

const auto SerializeLessEqualTestParams = ::testing::Combine(
    ::testing::ValuesIn(CommonTestUtils::combineParams(data.inputShapes)),
    ::testing::ValuesIn(data.inputsPrecisions),
    ::testing::Values(data.opType),
    ::testing::ValuesIn(data.secondInputTypes),
    ::testing::Values(data.ieInputPrecision),
    ::testing::Values(data.ieOutputPrecision),
    ::testing::Values(data.deviceName),
    ::testing::Values(data.additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs, ComparisonLayerTest, SerializeLessEqualTestParams, ComparisonLayerTest::getTestCaseName);
} // namespace
