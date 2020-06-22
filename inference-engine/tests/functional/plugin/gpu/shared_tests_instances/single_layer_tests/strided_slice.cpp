// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/strided_slice.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

InferenceEngine::Precision precision[] = {InferenceEngine::Precision::FP32, InferenceEngine::Precision::I64};

stridedSliceSpecificParams ss_only_test_cases[] = {
        stridedSliceSpecificParams({ 128, 1 }, { 0, 0, 0 }, { 0, 0, 0 }, { 1, 1, 1 },
                                { 0, 1, 1 }, { 0, 1, 1 },  { 1, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 }),
        stridedSliceSpecificParams({ 128, 1 }, { 0, 0, 0 }, { 0, 0, 0 }, { 1, 1, 1},
                                { 1, 0, 1 }, { 1, 0, 1 },  { 0, 1, 0 },  { 0, 0, 0 },  { 0, 0, 0 }),
        stridedSliceSpecificParams({ 1, 12, 100 }, { 0, -1, 0 }, { 0, 0, 0 }, { 1, 1, 1 },
                                { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 1, 0 },  { 0, 0, 0 }),
        stridedSliceSpecificParams({ 1, 12, 100 }, { 0, 9, 0 }, { 0, 11, 0 }, { 1, 1, 1 },
                                { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 }),
        stridedSliceSpecificParams({ 1, 12, 100 }, { 0, 1, 0 }, { 0, -1, 0 }, { 1, 1, 1 },
                                { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 }),
        stridedSliceSpecificParams({ 1, 12, 100 }, { 0, 9, 0 }, { 0, 8, 0 }, { 1, -1, 1 },
                                { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 }),
        stridedSliceSpecificParams({ 1, 12, 100 }, { 0, 9, 0 }, { 0, 7, 0 }, { -1, -1, -1 },
                                { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 }),
        stridedSliceSpecificParams({ 1, 12, 100 }, { 0, 7, 0 }, { 0, 9, 0 }, { -1, 1, -1 },
                                { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 }),
        stridedSliceSpecificParams({ 1, 12, 100 }, { 0, 4, 0 }, { 0, 9, 0 }, { -1, 2, -1 },
                                { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 }),
        stridedSliceSpecificParams({ 1, 12, 100 }, { 0, 4, 0 }, { 0, 10, 0 }, { -1, 2, -1 },
                                { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 }),
        stridedSliceSpecificParams({ 1, 12, 100 }, { 0, 9, 0 }, { 0, 4, 0 }, { -1, -2, -1 },
                                { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 }),
        stridedSliceSpecificParams({ 1, 12, 100 }, { 0, 10, 0 }, { 0, 4, 0 }, { -1, -2, -1 },
                                { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 }),
        stridedSliceSpecificParams({ 1, 12, 100 }, { 0, 11, 0 }, { 0, 0, 0 }, { -1, -2, -1 },
                                { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 }),
        stridedSliceSpecificParams({ 1, 12, 100 }, { 0, -6, 0 }, { 0, -8, 0 }, { -1, -2, -1 },
                                { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 }),
        stridedSliceSpecificParams({ 1, 12, 100, 1, 1 }, { 0, -1, 0, 0 }, { 0, 0, 0, 0 }, { 1, 1, 1, 1 },
                                { 1, 0, 1, 0 }, { 1, 0, 1, 0 },  { },  { 0, 1, 0, 1 },  {}),
        stridedSliceSpecificParams({ 2, 2, 2, 2 }, { 0, 0, 0, 0 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                                {1, 1, 1, 1}, {1, 1, 1, 1},  {},  {},  {}),
        stridedSliceSpecificParams({ 2, 2, 2, 2 }, { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                                {0, 0, 0, 0}, {1, 1, 1, 1},  {},  {},  {}),
        stridedSliceSpecificParams({ 2, 2, 2, 2 }, { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                                {0, 0, 0, 0}, {0, 0, 0, 0},  {},  {},  {}),
        stridedSliceSpecificParams({ 2, 2, 4, 3 }, { 0, 0, 0, 0 }, { 2, 2, 4, 3 }, { 1, 1, 2, 1 },
                                {1, 1, 1, 1}, {1, 1, 1, 1},  {},  {},  {}),
        stridedSliceSpecificParams({ 2, 2, 4, 2 }, { 1, 0, 0, 1 }, { 2, 2, 4, 2 }, { 1, 1, 2, 1 },
                                {0, 1, 1, 0}, {1, 1, 0, 0},  {},  {},  {}),
        stridedSliceSpecificParams({ 1, 2, 4, 2 }, { 1, 0, 0, 0 }, { 1, 2, 4, 2 }, { 1, 1, -2, -1 },
                                {1, 1, 1, 1}, {1, 1, 1, 1},  {},  {},  {}),
        stridedSliceSpecificParams({ 2, 2, 4, 2 }, { 1, 0, 0, 0 }, { 1, 2, 4, 2 }, { 1, 1, -2, -1 },
                                {0, 1, 1, 1}, {1, 1, 1, 1},  {},  {},  {}),
        stridedSliceSpecificParams({ 2, 2, 2, 2 }, { 0, 0, 0, 0 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                                {1, 1, 1, 1}, {1, 1, 1, 1},  {},  {},  {}),
        stridedSliceSpecificParams({ 2, 2, 2, 2 }, { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                                {0, 0, 0, 0}, {1, 1, 1, 1},  {},  {},  {}),
        stridedSliceSpecificParams({ 2, 2, 2, 2 }, { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                                {0, 0, 0, 0}, {0, 0, 0, 0},  {},  {},  {}),
        stridedSliceSpecificParams({ 2, 3, 4, 5, 6 }, { 0, 1, 0, 0, 0 }, { 2, 3, 4, 5, 6 }, { 1, 1, 1, 1, 1 },
                                {1, 0, 1, 1, 1}, {1, 0, 1, 1, 1},  {},  {0, 1, 0, 0, 0},  {}),
};

INSTANTIATE_TEST_CASE_P(
        smoke_CLDNN, StridedSliceLayerTest,
                ::testing::Combine(
                        ::testing::ValuesIn(ss_only_test_cases),
                        ::testing::ValuesIn(precision),
                        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
        StridedSliceLayerTest::getTestCaseName);


}  // namespace