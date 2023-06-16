// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/split_concat_memory.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

const std::vector<ov::element::Type_t> netPrecisions = {
        ov::element::Type_t::f32,
        ov::element::Type_t::i32,
        ov::element::Type_t::f16,
        ov::element::Type_t::i16,
        ov::element::Type_t::u8,
        ov::element::Type_t::i8,
};

const std::vector<InferenceEngine::SizeVector> shapes = {
    {1, 8, 3, 2},
    {3, 8, 3, 2},
    {3, 8, 3},
    {3, 8},
};

INSTANTIATE_TEST_SUITE_P(smoke_CPU, SplitConcatMemory,
                        ::testing::Combine(
                                ::testing::ValuesIn(shapes),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(1),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        SplitConcatMemory::getTestCaseName);
}  // namespace
