// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/multiply_add.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

INSTANTIATE_TEST_CASE_P(MultipleAdd, MultiplyAddLayerTest,
                        ::testing::Combine(
                                ::testing::Values(std::vector<size_t >({1, 3, 2, 2, 4, 5})),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        MultiplyAddLayerTest::getTestCaseName);

}  // namespace
