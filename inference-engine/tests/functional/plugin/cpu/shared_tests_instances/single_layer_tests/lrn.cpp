// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/lrn.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
// Common params

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const double alpha = 9.9e-05;
const size_t beta = 2;
const size_t bias = 1.0f;
const size_t size = 5;

INSTANTIATE_TEST_CASE_P(LrnCheck, LrnLayerTest,
                        ::testing::Combine(::testing::Values(alpha),
                                           ::testing::Values(beta),
                                           ::testing::Values(bias),
                                           ::testing::Values(size),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(std::vector<size_t>({10, 10, 3, 2})),
                                           ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        LrnLayerTest::getTestCaseName);

}  // namespace
