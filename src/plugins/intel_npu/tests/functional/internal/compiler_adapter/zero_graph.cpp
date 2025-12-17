// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_graph.hpp"

namespace {
const std::vector<ov::AnyMap> configsGraphCompilationTests = {{},
                                                              {ov::cache_dir("test")},
                                                              {ov::intel_npu::bypass_umd_caching(true)}};

// tested versions interval is [1.5, CURRENT + 1)
auto graphExtVersions = ::testing::Range(ZE_MAKE_VERSION(1, 5), ZE_GRAPH_EXT_VERSION_CURRENT + 1);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         ZeroGraphCompilationTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configsGraphCompilationTests),
                                            graphExtVersions),
                         ZeroGraphTest::getTestCaseName);

const std::vector<ov::AnyMap> emptyConfigsTests = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         ZeroGraphTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(emptyConfigsTests),
                                            graphExtVersions),
                         ZeroGraphTest::getTestCaseName);
}  // namespace
