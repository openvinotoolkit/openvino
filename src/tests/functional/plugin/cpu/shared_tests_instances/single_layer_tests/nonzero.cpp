// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/nonzero.hpp"
#include "common_test_utils/test_constants.hpp"
#include <vector>

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {
    std::vector<ov::test::InputShape> inShapes = {
        {{}, {{1000}}},
        {{}, {{4, 1000}}},
        {{}, {{2, 4, 1000}}},
        {{}, {{2, 4, 4, 1000}}},
        {{}, {{2, 4, 4, 2, 1000}}},
    };

    const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::U8,
    };

    ConfigMap config;

    INSTANTIATE_TEST_SUITE_P(smoke_nonzero, NonZeroLayerTest,
                            ::testing::Combine(
                                ::testing::ValuesIn(inShapes),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::Values(config)),
                            NonZeroLayerTest::getTestCaseName);
} // namespace
