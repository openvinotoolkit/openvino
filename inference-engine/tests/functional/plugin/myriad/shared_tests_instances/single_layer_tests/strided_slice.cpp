// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/strided_slice.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

InferenceEngine::Precision precisions[] = {InferenceEngine::Precision::FP32, InferenceEngine::Precision::I32};

stridedSliceSpecificParams stridedSliceTestCases[] = {
        stridedSliceSpecificParams({10}, {0}, {10}, {2}, {0}, {0}, {0}, {0}, {0}),
        stridedSliceSpecificParams({10}, {1}, {9}, {2}, {0}, {0}, {0}, {0}, {0}),
        stridedSliceSpecificParams({1000, 4}, {0, 0}, {1000, 4}, {1, 4}, {0, 1}, {0, 1}, {0, 0}, {0, 0}, {0, 0}),
        stridedSliceSpecificParams({1, 2, 35, 33}, {0, 0, 0, 2}, {1, 2, 33, 31}, {1, 1, 1, 2}, {0, 0, 0, 0}, {0, 0, 0, 0},
                                   {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}),
        stridedSliceSpecificParams({2, 2, 2, 3}, {0, 0, 0, 1}, {2, 2, 2, 3}, {1, 2, 2, 2}, {1, 1, 0, 1},  {1, 1, 0, 1},
                                   {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}),
        stridedSliceSpecificParams({2, 8, 32, 32}, {0, 2, 0, 0}, {2, 7, 32, 32}, {1, 3, 1, 1}, {0, 0, 0, 0}, {0, 0, 0, 0},
                                   {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}),
        stridedSliceSpecificParams({2, 8, 32, 32}, {0, 0, 2, 0}, {2, 8, 31, 32}, {1, 1, 3, 1}, {0, 0, 0, 0}, {0, 0, 0, 0},
                                   {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}),
        stridedSliceSpecificParams({2, 8, 32, 32}, {0, 0, 0, 2}, {2, 8, 32, 32}, {1, 1, 1, 3}, {0, 0, 0, 0}, {0, 0, 0, 0},
                                   {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}),
        stridedSliceSpecificParams({1, 32, 128, 128}, {0, 0, 0, 0}, {1, 32, 128, 128}, {1, 2, 4, 8}, {0, 0, 0, 0}, {0, 0, 0, 0},
                                   {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}),
        stridedSliceSpecificParams({1, 32, 128, 128}, {0, 16, 0, 0}, {1, 32, 128, 128}, {1, 2, 4, 8}, {0, 0, 0, 0}, {0, 0, 0, 0},
                                   {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}),
        stridedSliceSpecificParams({4, 1000}, {0, 0}, {4, 9999}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, {0, 0}),
        stridedSliceSpecificParams({4, 1000}, {0, 0}, {4, -1}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, {0, 0}),
        stridedSliceSpecificParams({4, 1000}, {0, 0}, {4, -3}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, {0, 0})
};

INSTANTIATE_TEST_CASE_P(
        smoke_Myriad, StridedSliceLayerTest,
                ::testing::Combine(
                        ::testing::ValuesIn(stridedSliceTestCases),
                        ::testing::ValuesIn(precisions),
                        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
        StridedSliceLayerTest::getTestCaseName);

}  // namespace