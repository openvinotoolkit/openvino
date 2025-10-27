// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_graph.hpp"

#include <common_test_utils/test_assertions.hpp>

namespace {
std::vector<int> graphDescflags = {ZE_GRAPH_FLAG_NONE, ZE_GRAPH_FLAG_DISABLE_CACHING, ZE_GRAPH_FLAG_ENABLE_PROFILING};

// tested versions interval is [1.5, CURRENT + 1)
auto extVersions = ::testing::Range(ZE_MAKE_VERSION(1, 5), ZE_GRAPH_EXT_VERSION_CURRENT + 1);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         ZeroGraphCompilationTests,
                         ::testing::Combine(::testing::ValuesIn(graphDescflags), extVersions),
                         ZeroGraphTest::getTestCaseName);

std::vector<int> noneGraphDescflags = {ZE_GRAPH_FLAG_NONE};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         ZeroGraphTest,
                         ::testing::Combine(::testing::ValuesIn(noneGraphDescflags), extVersions),
                         ZeroGraphTest::getTestCaseName);
}  // namespace
