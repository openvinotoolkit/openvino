// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/plugin/hetero_synthetic.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/subgraph_builders.hpp"

namespace {
using namespace HeteroTests;

// this tests load plugin by library name: this is not available during static linkage
#ifndef OPENVINO_STATIC_LIBRARY

INSTANTIATE_TEST_SUITE_P(
    smoke_manyTargetInputs,
    HeteroSyntheticTest,
    ::testing::Combine(::testing::Values(std::vector<PluginParameter>{{"TEMPLATE0", "openvino_template_plugin"},
                                                                      {"TEMPLATE1", "openvino_template_plugin"}}),
                       ::testing::ValuesIn(HeteroTests::HeteroSyntheticTest::withMajorNodesFunctions(
                           [] {
                               return ngraph::builder::subgraph::makeConvPool2Relu2();
                           },
                           {"Conv_1"},
                           true))),
    HeteroSyntheticTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_SingleMajorNode,
    HeteroSyntheticTest,
    ::testing::Combine(::testing::Values(std::vector<PluginParameter>{{"TEMPLATE0", "openvino_template_plugin"},
                                                                      {"TEMPLATE1", "openvino_template_plugin"}}),
                       ::testing::ValuesIn(HeteroTests::HeteroSyntheticTest::_singleMajorNodeFunctions)),
    HeteroSyntheticTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    nightly_RandomMajorNodes,
    HeteroSyntheticTest,
    ::testing::Combine(::testing::Values(std::vector<PluginParameter>{{"TEMPLATE0", "openvino_template_plugin"},
                                                                      {"TEMPLATE1", "openvino_template_plugin"}}),
                       ::testing::ValuesIn(HeteroTests::HeteroSyntheticTest::_randomMajorNodeFunctions)),
    HeteroSyntheticTest::getTestCaseName);

static std::vector<std::function<std::shared_ptr<ov::Model>()>> dynamicBuilders = {
    [] {
        return ngraph::builder::subgraph::makeConvPoolReluNonZero();
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_NonZeroMajorNode_dynamic,
    HeteroSyntheticTest,
    ::testing::Combine(
        ::testing::Values(std::vector<PluginParameter>{{"TEMPLATE0", "openvino_template_plugin"},
                                                       {"TEMPLATE1", "openvino_template_plugin"}}),
        ::testing::ValuesIn(HeteroTests::HeteroSyntheticTest::withMajorNodesFunctions(dynamicBuilders.front(),
                                                                                      {"nonZero_1"}))),
    HeteroSyntheticTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_NonZeroMajorNode_dynamic_batch,
    HeteroSyntheticTest,
    ::testing::Combine(
        ::testing::Values(std::vector<PluginParameter>{{"TEMPLATE0", "openvino_template_plugin"},
                                                       {"TEMPLATE1", "openvino_template_plugin"}}),
        ::testing::ValuesIn(
            HeteroTests::HeteroSyntheticTest::withMajorNodesFunctions(dynamicBuilders.front(), {"nonZero_1"}, true))),
    HeteroSyntheticTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_SingleMajorNode_dynamic,
    HeteroSyntheticTest,
    ::testing::Combine(
        ::testing::Values(std::vector<PluginParameter>{{"TEMPLATE0", "openvino_template_plugin"},
                                                       {"TEMPLATE1", "openvino_template_plugin"}}),
        ::testing::ValuesIn(HeteroTests::HeteroSyntheticTest::singleMajorNodeFunctions(dynamicBuilders))),
    HeteroSyntheticTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    nightly_RandomMajorNodes_dynamic,
    HeteroSyntheticTest,
    ::testing::Combine(
        ::testing::Values(std::vector<PluginParameter>{{"TEMPLATE0", "openvino_template_plugin"},
                                                       {"TEMPLATE1", "openvino_template_plugin"}}),
        ::testing::ValuesIn(HeteroTests::HeteroSyntheticTest::randomMajorNodeFunctions(dynamicBuilders))),
    HeteroSyntheticTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_SingleMajorNode_dynamic_batch,
    HeteroSyntheticTest,
    ::testing::Combine(::testing::Values(std::vector<PluginParameter>{{"TEMPLATE0", "openvino_template_plugin"},
                                                                      {"TEMPLATE1", "openvino_template_plugin"}}),
                       ::testing::ValuesIn(HeteroTests::HeteroSyntheticTest::singleMajorNodeFunctions(dynamicBuilders,
                                                                                                      true))),
    HeteroSyntheticTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    nightly_RandomMajorNodes_dynamic_batch,
    HeteroSyntheticTest,
    ::testing::Combine(::testing::Values(std::vector<PluginParameter>{{"TEMPLATE0", "openvino_template_plugin"},
                                                                      {"TEMPLATE1", "openvino_template_plugin"}}),
                       ::testing::ValuesIn(HeteroTests::HeteroSyntheticTest::randomMajorNodeFunctions(dynamicBuilders,
                                                                                                      true))),
    HeteroSyntheticTest::getTestCaseName);

#endif  // !OPENVINO_STATIC_LIBRARY

}  // namespace
