// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/hetero_synthetic.hpp"

#include <vector>

#include "openvino/core/visibility.hpp"

#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
const char* cpu_plugin_file_name = "openvino_arm_cpu_plugin";
#elif defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
const char* cpu_plugin_file_name = "openvino_intel_cpu_plugin";
#elif defined(OPENVINO_ARCH_RISCV64)
const char* cpu_plugin_file_name = "openvino_riscv_cpu_plugin";
#else
#    error "Undefined system processor"
#endif

namespace {
using ov::test::behavior::OVHeteroSyntheticTest;
using ov::test::behavior::PluginParameter;

// this tests load plugin by library name: this is not available during static linkage
#ifndef OPENVINO_STATIC_LIBRARY

INSTANTIATE_TEST_SUITE_P(nightly_SingleMajorNode,
                         OVHeteroSyntheticTest,
                         ::testing::Combine(::testing::Values(std::vector<PluginParameter>{
                                                {"CPU0", cpu_plugin_file_name},
                                                {"CPU1", cpu_plugin_file_name}}),
                                            ::testing::ValuesIn(OVHeteroSyntheticTest::_singleMajorNodeFunctions)),
                         OVHeteroSyntheticTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_RandomMajorNodes,
                         OVHeteroSyntheticTest,
                         ::testing::Combine(::testing::Values(std::vector<PluginParameter>{
                                                {"CPU0", cpu_plugin_file_name},
                                                {"CPU1", cpu_plugin_file_name}}),
                                            ::testing::ValuesIn(OVHeteroSyntheticTest::_randomMajorNodeFunctions)),
                         OVHeteroSyntheticTest::getTestCaseName);

#endif  // !OPENVINO_STATIC_LIBRARY

}  // namespace
