// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/memory_fq_concat_prelu.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "gna/gna_config.hpp"

namespace SubgraphTestsDefinitions {
namespace {

std::vector<std::vector<std::vector<size_t>>> inputs{{{1, 64}}};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}},
                                                           {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

std::vector<std::tuple<std::vector<int64_t>,
                       std::vector<int64_t>,
                       std::vector<int64_t>,
                       std::vector<int64_t>,
                       std::vector<int64_t>>>
    strided_slice_params = {std::make_tuple(std::vector<int64_t>{0, 0},
                                            std::vector<int64_t>{1, 64},
                                            std::vector<int64_t>{1, 1},
                                            std::vector<int64_t>{1, 0},
                                            std::vector<int64_t>{1, 0})};

std::vector<std::tuple<std::size_t,
                       std::vector<size_t>,
                       std::vector<float>,
                       std::vector<float>,
                       std::vector<float>,
                       std::vector<float>>>
    fake_quantize_params = {std::make_tuple(65535,
                                            std::vector<size_t>{1},
                                            std::vector<float>{-1},
                                            std::vector<float>{1},
                                            std::vector<float>{-1},
                                            std::vector<float>{1})};

}  // namespace

INSTANTIATE_TEST_SUITE_P(smoke_memory_fq_concat_prelu,
                         MemoryFqConcatPrelu,
                         ::testing::Combine(::testing::ValuesIn(inputs),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(strided_slice_params),
                                            ::testing::ValuesIn(fake_quantize_params)),
                         MemoryFqConcatPrelu::getTestCaseName);
}  // namespace SubgraphTestsDefinitions
