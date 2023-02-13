// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_integration.hpp"
#include "behavior/ov_plugin/core_integration_sw_plugin.hpp"
#include "behavior/ov_plugin/query_model.hpp"
#include "behavior/ov_plugin/compile_model.hpp"
#include "behavior/ov_plugin/properties_tests.hpp"
#include "openvino/runtime/core.hpp"
#include "ov_api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;
using namespace InferenceEngine::PluginConfigParams;

namespace {
//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassBasicTestP,
        ::testing::ValuesIn(generate_pairs_plugin_name_by_device()));

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassNetworkTestP,
        ::testing::ValuesIn(return_all_possible_device_combination()));

//
// IE Class GetMetric
//
INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::ValuesIn(return_all_possible_device_combination(false)));

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassGetMetricTest_RANGE_FOR_STREAMS,
        ::testing::ValuesIn(return_all_possible_device_combination(false)));

std::vector<ov::AnyMap> incorrect_properies = {{{"unsupported_key", "4"}}};

INSTANTIATE_TEST_SUITE_P(
        smoke_OVGetPropertiesIncorrectTests, OVGetPropertiesIncorrectTests,
        ::testing::Combine(::testing::ValuesIn(return_all_possible_device_combination(false)),
                           ::testing::ValuesIn(incorrect_properies)));

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassGetAvailableDevices,
        ::testing::ValuesIn(return_all_possible_device_combination(false)));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassGetConfigTest,
        ::testing::ValuesIn(return_all_possible_device_combination(false)));

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassQueryNetworkTest,
        ::testing::ValuesIn(return_all_possible_device_combination(false)));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassLoadNetworkTest,
        ::testing::Values(targetDevice));
} // namespace

