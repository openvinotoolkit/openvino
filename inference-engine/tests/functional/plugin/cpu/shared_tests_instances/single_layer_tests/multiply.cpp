// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/multiply.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

    std::vector<std::vector<std::vector<size_t>>> inShapes = {
            {{2}},
            {{1, 1, 1, 3}},
            {{1, 2, 4}},
            {{1, 4, 4}},
            {{1, 4, 4, 1}},
            {{1, 1, 1, 1, 1, 1, 3}},
            {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}
    };

    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                             InferenceEngine::Precision::FP16,
    };

    INSTANTIATE_TEST_CASE_P(multilpy, MultiplyLayerTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(inShapes),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MultiplyLayerTest::getTestCaseName);
}  // namespace
