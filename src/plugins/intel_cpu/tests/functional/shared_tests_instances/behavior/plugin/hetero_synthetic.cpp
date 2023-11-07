// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/plugin/hetero_synthetic.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/subgraph_builders.hpp"

// defined in plugin_name.cpp
extern const char * cpu_plugin_file_name;

namespace {
using namespace HeteroTests;

// this tests load plugin by library name: this is not available during static linkage
#ifndef OPENVINO_STATIC_LIBRARY

INSTANTIATE_TEST_SUITE_P(smoke_SingleMajorNode, HeteroSyntheticTest,
                        ::testing::Combine(
                                ::testing::Values(std::vector<PluginParameter>{{"CPU0", cpu_plugin_file_name}, {"CPU1", cpu_plugin_file_name}}),
                                ::testing::ValuesIn(HeteroTests::HeteroSyntheticTest::_singleMajorNodeFunctions)),
                        HeteroSyntheticTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_RandomMajorNodes, HeteroSyntheticTest,
                        ::testing::Combine(
                                ::testing::Values(std::vector<PluginParameter>{{"CPU0", cpu_plugin_file_name}, {"CPU1", cpu_plugin_file_name}}),
                                ::testing::ValuesIn(HeteroTests::HeteroSyntheticTest::_randomMajorNodeFunctions)),
                        HeteroSyntheticTest::getTestCaseName);

#endif // !OPENVINO_STATIC_LIBRARY

}  // namespace
