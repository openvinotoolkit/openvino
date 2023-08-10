// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/first_connect_input_concat.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

std::vector<std::vector<std::vector<size_t>>> inShapes = {{{1, 1288}}, {{1, 56}}, {{1, 6456}}};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

std::vector<std::map<std::string, std::string>> additional_config = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_COMPACT_MODE", "NO"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}, {"GNA_COMPACT_MODE", "NO"}}};

INSTANTIATE_TEST_SUITE_P(smoke_concat_first_input,
                         ConcatFirstInputTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(additional_config)),
                         ConcatFirstInputTest::getTestCaseName);

}  // namespace
