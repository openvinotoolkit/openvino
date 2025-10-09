// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_graph.hpp"

#include <common_test_utils/test_assertions.hpp>

namespace {
std::vector<int> graphDescflags = {ZE_GRAPH_FLAG_NONE,
                                   ZE_GRAPH_FLAG_DISABLE_CACHING,
                                   ZE_GRAPH_FLAG_ENABLE_PROFILING,
                                   ZE_GRAPH_FLAG_INPUT_GRAPH_PERSISTENT};

auto extVersions = ::testing::Range(ZE_MAKE_VERSION(1, 5), ZE_GRAPH_EXT_VERSION_CURRENT + 1);

INSTANTIATE_TEST_SUITE_P(something,
                         ZeroGraphTest,
                         ::testing::Combine(::testing::ValuesIn(graphDescflags), extVersions),
                         ZeroGraphTest::getTestCaseName);
}  // namespace
