// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <common_test_utils/test_constants.hpp>
#include <ie_core.hpp>
#include <ie_metric_helpers.hpp>
#include <multi-device/multi_device_config.hpp>
#include <ngraph_functions/subgraph_builders.hpp>

#include "cpp/ie_plugin.hpp"
#include "mock_common.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "plugin/mock_auto_device_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"

using ::testing::_;
using ::testing::AllOf;
using ::testing::AnyNumber;
using ::testing::AtLeast;
using ::testing::ContainsRegex;
using ::testing::Eq;
using ::testing::MatcherCast;
using ::testing::Matches;
using ::testing::NiceMock;
using ::testing::Property;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrEq;
using ::testing::Throw;
using Config = std::map<std::string, std::string>;
using namespace MockMultiDevice;

using ConfigParams = std::tuple<std::string,  // target device name
                                ov::AnyMap    // configuration
                                >;

class SetPropertyThroughAuto : public ::testing::TestWithParam<ConfigParams> {
public:
    std::shared_ptr<ov::Core> core;
    std::shared_ptr<NiceMock<MockMultiDeviceInferencePlugin>> plugin;
    std::shared_ptr<ngraph::Function> actualNetwork;
    std::string device;
    ov::AnyMap configuration;
    void TearDown() override {
        core.reset();
        plugin.reset();
        actualNetwork.reset();
    }
    void SetUp() override {
        std::tie(device, configuration) = this->GetParam();
        core = std::make_shared<ov::Core>();
        NiceMock<MockMultiDeviceInferencePlugin>* mock_multi = new NiceMock<MockMultiDeviceInferencePlugin>();
        plugin.reset(mock_multi);
        // Generic network
        actualNetwork = ngraph::builder::subgraph::makeSplitConvConcat();
    }
};
using AutoSetPropertyIncorrectTest = SetPropertyThroughAuto;
using AutoLoadNetworkCorrectTest = SetPropertyThroughAuto;
using AutoLoadNetworkIncorrectTest = SetPropertyThroughAuto;

const std::vector<ov::AnyMap> configsForSetPropertyCorrectTest = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT), ov::device::priorities("CPU")},
    {ov::hint::allow_auto_batching(false), ov::device::priorities("CPU")},
    {ov::enable_profiling(true), ov::device::priorities("CPU")}};

const std::vector<ov::AnyMap> configsForAutoSetPropertyCorrectTest = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
    {ov::hint::allow_auto_batching(false)},
    {ov::enable_profiling(true)}};

const std::vector<ov::AnyMap> configsForSetUnsupportedPropertyIncorrectTest = {
    {ov::num_streams(4), ov::device::priorities("CPU")}};
const std::vector<ov::AnyMap> configsAutoForSetUnsupportedPropertyIncorrectTest = {{ov::num_streams(4)}};

const std::vector<ov::AnyMap> configsForSetPropertyWithDevicePropertiesTest = {
    {ov::device::properties("CPU", ov::num_streams(10)), ov::device::priorities("CPU")},
    {ov::enable_profiling(true), ov::device::properties("CPU", ov::num_streams(10)), ov::device::priorities("CPU")}};

const std::vector<ov::AnyMap> configsAutoForSetPropertyWithDevicePropertiesTest = {
    {ov::device::properties("CPU", ov::num_streams(10))},
    {ov::enable_profiling(true), ov::device::properties("CPU", ov::num_streams(10))}};

TEST_P(AutoLoadNetworkCorrectTest, smoke_AUTO_loadNetworkWithCorrectPropertyTestNoThrow) {
    ASSERT_NO_THROW(core->compile_model(actualNetwork, device, configuration));
}

TEST_P(AutoLoadNetworkCorrectTest, smoke_AUTO_loadNetworkWithSetPropertyFirstTestNoThrow) {
    ASSERT_NO_THROW(core->set_property(device, configuration));
    ov::AnyMap config = {};
    if (device == "MULTI")
        config = {ov::device::priorities("CPU")};
    ASSERT_NO_THROW(core->compile_model(actualNetwork, device, config));
}

TEST_P(AutoLoadNetworkIncorrectTest, smoke_AUTO_loadNetworkWithUnsupportedPropertyTestThrow) {
    ASSERT_THROW(core->compile_model(actualNetwork, device, configuration), ov::Exception);
}

TEST_P(AutoLoadNetworkIncorrectTest, smoke_AUTO_LoadNetworkWithSetPropertyFirstWithIncorrectConfigTestThrow) {
    ASSERT_NO_THROW(core->set_property(device, configuration));
    ov::AnyMap config = {};
    if (device == "MULTI")
        config = {ov::device::priorities("CPU")};
    ASSERT_THROW(core->compile_model(actualNetwork, device, config), ov::Exception);
}

TEST_P(AutoSetPropertyIncorrectTest, smoke_AUTO_LoadNetworkFirstWithSetPropertyWithIncorrectConfigTestNoThrow) {
    // Create plugin with empty configuration first
    ov::AnyMap config = {};
    if (device == "MULTI")
        config = {ov::device::priorities("CPU")};
    ASSERT_NO_THROW(core->compile_model(actualNetwork, device, config));
    ASSERT_THROW(core->set_property(device, configuration), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(smoke_Auto_loadNetworkWithSetPropertyBehaviorTests,
                         AutoLoadNetworkCorrectTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(configsForAutoSetPropertyCorrectTest)),
                         ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_Multi_loadNetworkWithSetPropertyBehaviorTests,
                         AutoLoadNetworkCorrectTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(configsForSetPropertyCorrectTest)),
                         ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_Multi_loadNetworkWithSetIncorrectPropertyBehaviorTests,
                         AutoLoadNetworkIncorrectTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(configsForSetUnsupportedPropertyIncorrectTest)),
                         ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_Auto_loadNetworkWithSetIncorrectPropertyBehaviorTests,
                         AutoLoadNetworkIncorrectTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(configsAutoForSetUnsupportedPropertyIncorrectTest)),
                         ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_Multi_PropertySettingBehaviorTests,
                         AutoSetPropertyIncorrectTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(configsForSetPropertyWithDevicePropertiesTest)),
                         ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_Auto_PropertySettingBehaviorTests,
                         AutoSetPropertyIncorrectTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(configsAutoForSetPropertyWithDevicePropertiesTest)),
                         ::testing::PrintToStringParamName());