// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/simple_if.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
    std::vector<std::vector<std::vector<size_t>>> inputShapes = {
            {{5, 7}, {5, 7}},
            {{30, 20, 10}, {30, 20, 10}}
    };

    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                             InferenceEngine::Precision::I8,
    };

    std::vector<bool> conditions = {true, false};

    INSTANTIATE_TEST_SUITE_P(smoke_If, SimpleIfTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(inputShapes),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::ValuesIn(conditions),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            SimpleIfTest::getTestCaseName);

    TEST_P(SimpleIfTest, CompareWithRefs) {
        Run();
    };

    INSTANTIATE_TEST_SUITE_P(smoke_If, SimpleIf2OutTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(inputShapes),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::ValuesIn(conditions),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            SimpleIf2OutTest::getTestCaseName);

    TEST_P(SimpleIf2OutTest, CompareWithRefs) {
        Run();
    };

    INSTANTIATE_TEST_SUITE_P(smoke_If, SimpleIfNotConstConditionTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(inputShapes),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::ValuesIn(conditions),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            SimpleIfNotConstConditionTest::getTestCaseName);

    TEST_P(SimpleIfNotConstConditionTest, CompareWithRefs) {
        Run();
    };

}  // namespace
