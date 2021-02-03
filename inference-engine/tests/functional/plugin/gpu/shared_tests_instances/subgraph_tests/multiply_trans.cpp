// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/multiply_trans.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<std::vector<size_t>>> cases = {
    {{1, 62, 12, 18, 510}, {0, 2, 3, 4, 1}},
    {{1, 64, 12, 18, 510}, {0, 2, 3, 4, 1}},
    {{1, 64, 4, 12, 18, 512}, {0, 2, 3, 4, 5, 1}},
    {{1, 64, 12, 18, 512}, {0, 2, 3, 4, 1}},
    {{1, 810, 64, 64}, {0, 2, 3, 1}}
};

INSTANTIATE_TEST_CASE_P(taylorMultiplyTrans_Nd, MultiplyTransLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(cases),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        MultiplyTransLayerTest::getTestCaseName);

}  // namespace
