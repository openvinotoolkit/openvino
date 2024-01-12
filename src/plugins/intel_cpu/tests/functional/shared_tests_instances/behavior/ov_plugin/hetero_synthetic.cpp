// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_plugin/hetero_synthetic.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/subgraph_builders.hpp"

// defined in plugin_name.cpp
extern const char * cpu_plugin_file_name;

namespace {
using ov::test::behavior::OVHeteroSyntheticTest;
using ov::test::behavior::PluginParameter;

// this tests load plugin by library name: this is not available during static linkage
#ifndef OPENVINO_STATIC_LIBRARY

INSTANTIATE_TEST_SUITE_P(smoke_SingleMajorNode, OVHeteroSyntheticTest,
                        ::testing::Combine(
                                ::testing::Values(std::vector<PluginParameter>{{"CPU0", cpu_plugin_file_name}, {"CPU1", cpu_plugin_file_name}}),
                                ::testing::ValuesIn(OVHeteroSyntheticTest::_singleMajorNodeFunctions)),
                        OVHeteroSyntheticTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_RandomMajorNodes, OVHeteroSyntheticTest,
                        ::testing::Combine(
                                ::testing::Values(std::vector<PluginParameter>{{"CPU0", cpu_plugin_file_name}, {"CPU1", cpu_plugin_file_name}}),
                                ::testing::ValuesIn(OVHeteroSyntheticTest::_randomMajorNodeFunctions)),
                        OVHeteroSyntheticTest::getTestCaseName);

#endif // !OPENVINO_STATIC_LIBRARY

}  // namespace
