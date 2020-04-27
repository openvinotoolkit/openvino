// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/strided_slice.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

stridedSliceParamsTuple ss_only_test_cases[] = {
        stridedSliceParamsTuple({ 2, 2, 2, 2 }, { 0, 0, 0, 0 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                       {1, 1, 1, 1}, {1, 1, 1, 1},  {1, 1, 1, 1},  {1, 1, 1, 1},  {1, 1, 1, 1},
                                InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_GPU),
        stridedSliceParamsTuple({ 2, 2, 2, 2 }, { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                       {0, 0, 0, 0}, {1, 1, 1, 1},  {1, 1, 1, 1},  {1, 1, 1, 1},  {1, 1, 1, 1},
                                InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_GPU),
        stridedSliceParamsTuple({ 2, 2, 2, 2 }, { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                       {0, 0, 0, 0}, {0, 0, 0, 0},  {1, 1, 1, 1},  {1, 1, 1, 1},  {1, 1, 1, 1},
                                InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_GPU),
        stridedSliceParamsTuple({ 2, 2, 4, 3 }, { 0, 0, 0, 0 }, { 2, 2, 4, 3 }, { 1, 1, 2, 1 },
                       {1, 1, 1, 1}, {1, 1, 1, 1},  {1, 1, 1, 1},  {1, 1, 1, 1},  {1, 1, 1, 1},
                                InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_GPU),
        stridedSliceParamsTuple({ 2, 2, 4, 2 }, { 1, 0, 0, 1 }, { 2, 2, 4, 2 }, { 1, 1, 2, 1 },
                       {0, 1, 1, 0}, {1, 1, 0, 0},  {1, 1, 1, 1},  {1, 1, 1, 1},  {1, 1, 1, 1},
                                InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_GPU),
        stridedSliceParamsTuple({ 1, 2, 4, 2 }, { 1, 0, 0, 0 }, { 1, 2, 4, 2 }, { 1, 1, -2, -1 },
                       {1, 1, 1, 1}, {1, 1, 1, 1},  {1, 1, 1, 1},  {1, 1, 1, 1},  {1, 1, 1, 1},
                                InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_GPU),
        stridedSliceParamsTuple({ 2, 2, 4, 2 }, { 1, 0, 0, 0 }, { 1, 2, 4, 2 }, { 1, 1, -2, -1 },
                       {0, 1, 1, 1}, {1, 1, 1, 1},  {1, 1, 1, 1},  {1, 1, 1, 1},  {1, 1, 1, 1},
                                InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_GPU),
};

INSTANTIATE_TEST_CASE_P(
        smoke_CLDNN, StridedSliceLayerTest, ::testing::ValuesIn(ss_only_test_cases),
        StridedSliceLayerTest::getTestCaseName);


}  // namespace
