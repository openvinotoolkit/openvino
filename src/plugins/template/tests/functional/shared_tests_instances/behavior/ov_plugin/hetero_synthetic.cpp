// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/hetero_synthetic.hpp"

#include <vector>

#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu_non_zero.hpp"

namespace {
using ov::test::behavior::OVHeteroSyntheticTest;
using ov::test::behavior::PluginParameter;

// this tests load plugin by library name: this is not available during static linkage
#ifndef OPENVINO_STATIC_LIBRARY

INSTANTIATE_TEST_SUITE_P(smoke_manyTargetInputs,
                         OVHeteroSyntheticTest,
                         ::testing::Combine(::testing::Values(std::vector<PluginParameter>{
                                                {"TEMPLATE0", "openvino_template_plugin"},
                                                {"TEMPLATE1", "openvino_template_plugin"}}),
                                            ::testing::ValuesIn(OVHeteroSyntheticTest::withMajorNodesFunctions(
                                                [] {
                                                    return ov::test::utils::make_conv_pool2_relu2();
                                                },
                                                {"Conv_1"},
                                                true))),
                         OVHeteroSyntheticTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SingleMajorNode,
                         OVHeteroSyntheticTest,
                         ::testing::Combine(::testing::Values(std::vector<PluginParameter>{
                                                {"TEMPLATE0", "openvino_template_plugin"},
                                                {"TEMPLATE1", "openvino_template_plugin"}}),
                                            ::testing::ValuesIn(OVHeteroSyntheticTest::_singleMajorNodeFunctions)),
                         OVHeteroSyntheticTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_RandomMajorNodes,
                         OVHeteroSyntheticTest,
                         ::testing::Combine(::testing::Values(std::vector<PluginParameter>{
                                                {"TEMPLATE0", "openvino_template_plugin"},
                                                {"TEMPLATE1", "openvino_template_plugin"}}),
                                            ::testing::ValuesIn(OVHeteroSyntheticTest::_randomMajorNodeFunctions)),
                         OVHeteroSyntheticTest::getTestCaseName);

static std::vector<std::function<std::shared_ptr<ov::Model>()>> dynamicBuilders = {
    [] {
        return ov::test::utils::make_conv_pool_relu_non_zero();
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_NonZeroMajorNode_dynamic,
    OVHeteroSyntheticTest,
    ::testing::Combine(::testing::Values(std::vector<PluginParameter>{{"TEMPLATE0", "openvino_template_plugin"},
                                                                      {"TEMPLATE1", "openvino_template_plugin"}}),
                       ::testing::ValuesIn(OVHeteroSyntheticTest::withMajorNodesFunctions(dynamicBuilders.front(),
                                                                                          {"nonZero_1"}))),
    OVHeteroSyntheticTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_NonZeroMajorNode_dynamic_batch,
    OVHeteroSyntheticTest,
    ::testing::Combine(::testing::Values(std::vector<PluginParameter>{{"TEMPLATE0", "openvino_template_plugin"},
                                                                      {"TEMPLATE1", "openvino_template_plugin"}}),
                       ::testing::ValuesIn(OVHeteroSyntheticTest::withMajorNodesFunctions(dynamicBuilders.front(),
                                                                                          {"nonZero_1"},
                                                                                          true))),
    OVHeteroSyntheticTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_SingleMajorNode_dynamic,
    OVHeteroSyntheticTest,
    ::testing::Combine(::testing::Values(std::vector<PluginParameter>{{"TEMPLATE0", "openvino_template_plugin"},
                                                                      {"TEMPLATE1", "openvino_template_plugin"}}),
                       ::testing::ValuesIn(OVHeteroSyntheticTest::singleMajorNodeFunctions(dynamicBuilders))),
    OVHeteroSyntheticTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    nightly_RandomMajorNodes_dynamic,
    OVHeteroSyntheticTest,
    ::testing::Combine(::testing::Values(std::vector<PluginParameter>{{"TEMPLATE0", "openvino_template_plugin"},
                                                                      {"TEMPLATE1", "openvino_template_plugin"}}),
                       ::testing::ValuesIn(OVHeteroSyntheticTest::randomMajorNodeFunctions(dynamicBuilders))),
    OVHeteroSyntheticTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_SingleMajorNode_dynamic_batch,
    OVHeteroSyntheticTest,
    ::testing::Combine(::testing::Values(std::vector<PluginParameter>{{"TEMPLATE0", "openvino_template_plugin"},
                                                                      {"TEMPLATE1", "openvino_template_plugin"}}),
                       ::testing::ValuesIn(OVHeteroSyntheticTest::singleMajorNodeFunctions(dynamicBuilders, true))),
    OVHeteroSyntheticTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    nightly_RandomMajorNodes_dynamic_batch,
    OVHeteroSyntheticTest,
    ::testing::Combine(::testing::Values(std::vector<PluginParameter>{{"TEMPLATE0", "openvino_template_plugin"},
                                                                      {"TEMPLATE1", "openvino_template_plugin"}}),
                       ::testing::ValuesIn(OVHeteroSyntheticTest::randomMajorNodeFunctions(dynamicBuilders, true))),
    OVHeteroSyntheticTest::getTestCaseName);

#endif  // !OPENVINO_STATIC_LIBRARY

}  // namespace
