// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/stridedslice_concat.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}},
                                                                 {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

const std::vector<StridedSliceParams> sliceParams = {
    {{8, 16}, {1, 16}, {2, 16}, {1, 1}, {0, 1}, {0, 1}},
    {{1, 16}, {1, 1}, {1, 2}, {1, 1}, {1, 0}, {1, 0}},
    {{1, 8, 16}, {1, 1, 16}, {1, 2, 16}, {1, 1, 1}, {1, 0, 1}, {1, 0, 1}},
    {{8, 25}, {3, 25}, {4, 25}, {1, 1}, {0, 1}, {0, 1}}};

INSTANTIATE_TEST_SUITE_P(smoke_SliceConcatTest,
                         SliceConcatTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(sliceParams)),
                         SliceConcatTest::getTestCaseName);
}  // namespace
