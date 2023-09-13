// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/permute_concat_concat_permute.hpp"

#include <vector>

using namespace SubgraphTestsDefinitions;

namespace {
std::vector<std::vector<size_t>> inputs1{{{1, 8}}, {{8, 1}}, {{16, 2}}, {{8, 2}}};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_SUITE_P(smoke_permute_concat_concat_permute,
                         PermuteConcatConcatPermute,
                         ::testing::Combine(::testing::ValuesIn(inputs1),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         PermuteConcatConcatPermute::getTestCaseName);

}  // namespace
