// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "custom/subgraph_tests/src/classes/concat_sdp.hpp"

namespace ov {
namespace test {
namespace {

const std::vector<std::vector<InputShape>> inputShapes = {
    // greedy search
    {
        {{1, 8, -1, 128}, {{1, 8, 10, 128}, {1, 8, 1, 128}, {1, 8, 1, 128}, {1, 8, 20, 128}, {1, 8, 1, 128}}},
        {{1, 8, -1, 128}, {{1, 8, 0, 128}, {1, 8, 10, 128}, {1, 8, 11, 128}, {1, 8, 12, 128}, {1, 8, 32, 128}}},
    },
    // beam search
    {
        {{-1, 8, -1, 128}, {{4, 8, 10, 128}, {4, 8, 1, 128}, {4, 8, 1, 128}, {4, 8, 1, 128}, {4, 8, 1, 128}}},
        {{-1, 8, -1, 128}, {{4, 8, 0, 128}, {4, 8, 10, 128}, {4, 8, 11, 128}, {4, 8, 12, 128}, {4, 8, 13, 128}}},
    },
    // big batch to check cvt_copy fast-path inside mha_single_token_kernel
    {
        {{-1, 8, -1, 128}, {{129, 8, 10, 128}, {129, 8, 1, 128}, {129, 8, 1, 128}, {129, 8, 1, 128}, {129, 8, 1, 128}}},
        {{-1, 8, -1, 128}, {{129, 8, 0, 128}, {129, 8, 10, 128}, {129, 8, 11, 128}, {129, 8, 12, 128}, {129, 8, 13, 128}}},
    },
};

const ov::AnyMap cfg_none{};
const ov::AnyMap cfg_u8_sym{
    {"KEY_CACHE_PRECISION", "u8"},
    {"VALUE_CACHE_PRECISION", "u8"},
};

INSTANTIATE_TEST_SUITE_P(smoke_ConcatSDPTest,
                         ConcatSDPTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::Values(cfg_none, cfg_u8_sym),
                                            ::testing::Values(true, false),
                                            ::testing::Values<int64_t>(8),
                                            ::testing::Values<int64_t>(8)),
                         ConcatSDPTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
