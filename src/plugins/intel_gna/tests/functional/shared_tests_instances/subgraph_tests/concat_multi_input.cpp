// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/concat_multi_input.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

std::vector<std::vector<std::vector<size_t>>> inShapes = {
    {{1, 8}, {1, 8}},
    {{1, 3}, {1, 3}, {1, 3}},
    {{1, 16}, {1, 16}, {1, 16}},
    {{1, 16}, {1, 16}, {1, 16}, {1, 16}},
    {{1, 32}, {1, 32}, {1, 32}, {1, 32}},
    {{1, 16}, {1, 32}, {1, 16}, {1, 32}, {1, 16}, {1, 32}},
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

std::vector<std::map<std::string, std::string>> configs = {{
                                                               {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                                                               {"GNA_COMPACT_MODE", "NO"},
                                                               {"GNA_SCALE_FACTOR_0", "2048"},
                                                               {"GNA_PRECISION", "I16"},
                                                           },
                                                           {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

INSTANTIATE_TEST_SUITE_P(smoke_concat_multi_input,
                         ConcatMultiInput,
                         ::testing::Combine(::testing::ValuesIn(inShapes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         ConcatMultiInput::getTestCaseName);

}  // namespace
