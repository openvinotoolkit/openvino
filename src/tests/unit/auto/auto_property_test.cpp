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

using SetUnsupportedPropertyTestP = SetPropertyThroughAuto;
using SetSupportedPropertyTestP = SetPropertyThroughAuto;
using LoadNetworkWithSupportedPropertyTestP = SetPropertyThroughAuto;
using LoadNetworkWithUnsupportedPropertyTestP = SetPropertyThroughAuto;

TEST(SetPropertyOverwriteTest, smoke_AUTO_SetPropertyOverwriteTestNoThrow) {
    GTEST_SKIP() << "Disabled test due to blocking." << std::endl;
    ov::Core ie;
    int32_t curValue = -1;
    auto actualNetwork = ngraph::builder::subgraph::makeSplitConvConcat();
    ASSERT_NO_THROW(ie.set_property(CommonTestUtils::DEVICE_CPU, {ov::num_streams(2)}));
    ASSERT_NO_THROW(ie.set_property(CommonTestUtils::DEVICE_AUTO, {ov::device::properties("CPU", ov::num_streams(4))}));
    ASSERT_NO_THROW(ie.compile_model(actualNetwork, CommonTestUtils::DEVICE_AUTO, {}));
    ASSERT_NO_THROW(curValue = ie.get_property(CommonTestUtils::DEVICE_CPU, ov::num_streams));
    EXPECT_EQ(curValue, 4);
}

TEST(SetPropertyOverwriteTest, smoke_MULTI_SetPropertyOverwriteTestNoThrow) {
    GTEST_SKIP() << "Disabled test due to blocking." << std::endl;
    ov::Core ie;
    int32_t curValue = -1;
    auto actualNetwork = ngraph::builder::subgraph::makeSplitConvConcat();
    ASSERT_NO_THROW(ie.set_property(CommonTestUtils::DEVICE_CPU, {ov::num_streams(2)}));
    ASSERT_NO_THROW(
        ie.set_property(CommonTestUtils::DEVICE_MULTI, {ov::device::properties("CPU", ov::num_streams(4))}));
    ASSERT_NO_THROW(ie.compile_model(actualNetwork, CommonTestUtils::DEVICE_MULTI, {ov::device::priorities("CPU")}));
    ASSERT_NO_THROW(curValue = ie.get_property(CommonTestUtils::DEVICE_CPU, ov::num_streams));
    EXPECT_EQ(curValue, 4);
}

TEST_P(SetSupportedPropertyTestP, setSupportedPropertyTestNoThrow) {
    ov::Core ie;
    ASSERT_NO_THROW(ie.set_property(device, configuration));
    ov::AnyMap config = {};
    if (device == "MULTI")
        config = {ov::device::priorities("CPU")};
    ASSERT_NO_THROW(ie.compile_model(actualNetwork, device, config));
}

TEST_P(SetUnsupportedPropertyTestP, setUnsupportedPropertyTestNoThrow) {
    ASSERT_NO_THROW(core->set_property(device, configuration));
    ov::AnyMap config = {};
    if (device == "MULTI")
        config = {ov::device::priorities("CPU")};
    ASSERT_THROW(core->compile_model(actualNetwork, device, config), ov::Exception);
}

TEST_P(LoadNetworkWithSupportedPropertyTestP, loadNetworkSupportedPropertyTestNoThrow) {
    if (device == "MULTI")
        ASSERT_NO_THROW(core->set_property(device, {ov::device::priorities("CPU")}));
    ASSERT_NO_THROW(core->compile_model(actualNetwork, device, configuration));
}

TEST_P(LoadNetworkWithUnsupportedPropertyTestP, smoke_Auto_Multi_LoadNetworkUnsupportedPropertyTestNoThrow) {
    if (device == "MULTI")
        ASSERT_NO_THROW(core->set_property(device, {ov::device::priorities("CPU")}));
    ASSERT_THROW(core->compile_model(actualNetwork, device, configuration), ov::Exception);
}

const std::vector<ov::AnyMap> configsSupportedPropertyTest = {
    {ov::enable_profiling(true)},
    {ov::enable_profiling(true), ov::device::properties("CPU", ov::num_streams(4))}};

const std::vector<ov::AnyMap> configsUnsupportedPropertyTest = {
    {ov::num_streams(4)},
    {ov::num_streams(4), ov::enable_profiling(true)},
    {ov::device::properties("INVALID", ov::num_streams(4))}};
const std::vector<ov::AnyMap> configsMultiLoadNetworkSupportedPropertyTest = {
    {ov::num_streams(4)},
    {ov::enable_profiling(true)},
    {ov::num_streams(4), ov::enable_profiling(true)},
    {ov::device::properties("CPU", ov::num_streams(4))},
    {ov::num_streams(4), ov::device::properties("CPU", ov::num_streams(4))}};
const std::vector<ov::AnyMap> configsAutoLoadNetworkUnsupportedPropertyTest = configsUnsupportedPropertyTest;

INSTANTIATE_TEST_SUITE_P(smoke_Auto_Multi_SetSupportedPropertyTestNoThrow,
                         SetSupportedPropertyTestP,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO,
                                                              CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(configsSupportedPropertyTest)),
                         ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_Auto_Multi_SetUnsupportedPropertyTestNoThrow,
                         SetUnsupportedPropertyTestP,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO,
                                                              CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(configsUnsupportedPropertyTest)),
                         ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_Multi_LoadNetworkSupportedPropertyTestNoThrow,
                         LoadNetworkWithSupportedPropertyTestP,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(configsMultiLoadNetworkSupportedPropertyTest)),
                         ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_Auto_LoadNetworkUnsupportedPropertyTestNoThrow,
                         LoadNetworkWithUnsupportedPropertyTestP,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(configsAutoLoadNetworkUnsupportedPropertyTest)),
                         ::testing::PrintToStringParamName());
