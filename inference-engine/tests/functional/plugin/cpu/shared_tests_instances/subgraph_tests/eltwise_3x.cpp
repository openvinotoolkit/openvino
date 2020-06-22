// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <vector>
#include "subgraph_tests/eltwise_3x.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using InferenceEngine::Precision;
namespace {

    std::vector<std::vector<std::vector<size_t>>> inputShapes {
            {
                {{1, 1, 2, 3}, {1, 1, 2, 3}, {1, 1, 2, 3}, {1, 1, 2, 3}},
                {{1, 48, 5, 6}, {1, 48, 1, 1}, {1, 48, 5, 6}, {1, 1, 5, 6}},
                {{1, 72, 28, 28}, {1, 72, 1, 1}, {1, 72, 1, 1}, {1, 72, 1, 1}},
                {{1, 2, 3}, {3}, {3}, {3}},
                {{1, 12, 5, 5}, {5, 5}, {12, 5, 5}, {1}},
                {{3, 12, 5, 5}, {1, 12, 5, 1}, {3, 1, 1, 1}, {3, 12, 5, 5}},
                {{1, 1, 1, 1}, {1, 12, 5, 1}, {3, 12, 1, 5}, {3, 12, 5, 1}},
                {{1, 1, 1, 1, 6}, {1, 12, 5, 1, 6}, {3, 12, 1, 5, 1}, {3, 12, 5, 1, 1}}
            }
    };

    std::vector<std::vector<InferenceEngine::Precision>> inputPrecisions = {
            { Precision::FP32, Precision::FP32, Precision::FP32, Precision::FP32 },
            { Precision::I32, Precision::I32, Precision::I32, Precision::I32 }
    };



    INSTANTIATE_TEST_CASE_P(Eltwise3x, Eltwise3x,
                            ::testing::Combine(
                                    ::testing::ValuesIn(inputShapes),
                                    ::testing::ValuesIn(inputPrecisions),
                                    ::testing::ValuesIn({false}),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                    Eltwise3x::getTestCaseName);

    std::vector<std::vector<std::vector<size_t>>> inputShapesFQ {
            {
                {{1, 1, 2, 3}, {1, 1, 2, 3}, {1, 1, 2, 3}, {1, 1, 2, 3}},
                {{1, 33, 5, 5}, {5, 5}, {33, 5, 5}, {1}},
                {{3, 12, 5, 5}, {1, 12, 5, 1}, {3, 1, 1, 1}, {3, 12, 5, 5}},
                {{1, 12, 1, 1}, {1, 12, 5, 1}, {3, 12, 1, 5}, {3, 12, 5, 1}},
                {{1, 12, 1, 1, 6}, {1, 12, 5, 1, 6}, {3, 12, 1, 5, 1}, {3, 12, 5, 1, 1}}
            }
    };

    std::vector<std::vector<InferenceEngine::Precision>> inputPrecisionsFQ {
            { Precision::FP32, Precision::FP32, Precision::FP32, Precision::FP32 }
    };

    INSTANTIATE_TEST_CASE_P(Eltwise3xFQ, Eltwise3x,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapesFQ),
                                ::testing::ValuesIn(inputPrecisionsFQ),
                                ::testing::ValuesIn({true}),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                Eltwise3x::getTestCaseName);
}  // namespace
