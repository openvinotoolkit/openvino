// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "subgraph_tests/memory_concat.hpp"
#include "common_test_utils/test_constants.hpp"
#include "gna/gna_config.hpp"

namespace SubgraphTestsDefinitions {
namespace {

std::vector<std::vector<std::vector<size_t>>> inputs{
        {{1, 64}}
};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

std::map<std::string, std::string> additional_config = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}
};

std::vector<
    std::tuple<
        std::vector<int64_t>,
        std::vector<int64_t>,
        std::vector<int64_t>,
        std::vector<int64_t>,
        std::vector<int64_t>>> strided_slice_params = {
    {{{0, 0}, {1, 64}, {1, 1}, {1, 0}, {1, 0}}}
};

std::vector<
    std::tuple<
        std::size_t,
        std::vector<size_t>,
        std::vector<float>,
        std::vector<float>,
        std::vector<float>,
        std::vector<float>>> fake_quantize_params = {
    {{65535, { 1 }, { -1 }, { 1 }, { -1 }, { 1 }}}
};

} // namespace

INSTANTIATE_TEST_SUITE_P(smoke_memory_concat, MemoryConcat,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputs),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::Values(additional_config),
                                ::testing::ValuesIn(strided_slice_params),
                                ::testing::ValuesIn(fake_quantize_params)),
                        MemoryConcat::getTestCaseName);
} // namespace SubgraphTestsDefinitions
