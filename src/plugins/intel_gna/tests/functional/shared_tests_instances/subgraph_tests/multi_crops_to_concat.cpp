// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <subgraph_tests/multi_crops_to_concat.hpp>

#include "common_test_utils/test_constants.hpp"

namespace SubgraphTestsDefinitions {
namespace {
const std::vector<std::vector<size_t>> input_shapes = {{1, 48}, {1, 64}};

const std::vector<std::vector<std::pair<int64_t, int64_t>>> offsets = {{{0, 16}, {33, 48}},
                                                                       {{17, 32}, {33, 48}},
                                                                       {{5, 14}, {17, 26}},
                                                                       {{1, 8}, {9, 16}, {17, 24}}};

const std::vector<std::map<std::string, std::string>> additional_config = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}},
                                                                           {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};
}  // namespace

INSTANTIATE_TEST_SUITE_P(smoke_multi_crop_to_concat,
                         MultiCropsToConcatTest,
                         ::testing::Combine(::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(offsets),
                                            ::testing::ValuesIn(additional_config)),
                         MultiCropsToConcatTest::getTestCaseName);
}  // namespace SubgraphTestsDefinitions
