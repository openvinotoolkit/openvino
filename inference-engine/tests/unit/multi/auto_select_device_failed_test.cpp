// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_metric_helpers.hpp>
#include <common_test_utils/test_constants.hpp>
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include <ie_core.hpp>
#include <multi-device/multi_device_config.hpp>
#include <ngraph_functions/subgraph_builders.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "plugin/mock_multi_device_plugin.hpp"
#include "cpp/ie_plugin.hpp"

using ::testing::MatcherCast;
using ::testing::AllOf;
using ::testing::Throw;
using ::testing::Matches;
using ::testing::_;
using ::testing::StrEq;
using ::testing::Return;
using ::testing::Property;
using ::testing::Eq;
using ::testing::ReturnRef;
using Config = std::map<std::string, std::string>;
using namespace MockMultiDevice;

#define IE_SET_METRIC(key, name,  ...)                                                            \
    typename ::InferenceEngine::Metrics::MetricType<::InferenceEngine::Metrics::key>::type name = \
        __VA_ARGS__;

class AutoLoadFailedTest: public ::testing::Test {
protected:
    std::shared_ptr<ngraph::Function> function;
    InferenceEngine::CNNNetwork cnnNet;
    std::shared_ptr<MockICore> core;
    std::shared_ptr<MockMultiDeviceInferencePlugin> plugin;

    //mock exeNetwork
    std::shared_ptr<MockIExecutableNetworkInternal> mockIExeNet;
    ov::runtime::SoPtr<IExecutableNetworkInternal>  mockExeNetwork;
    MockIInferencePlugin*                           mockIPlugin;
    InferenceEngine::InferencePlugin                mockPlugin;
    std::map<std::string, std::string>              config;
    std::vector<DeviceInformation>                  metaDevices;

    void TearDown() override {
        core.reset();
        plugin.reset();
        mockIExeNet.reset();
        mockExeNetwork = {};
        mockPlugin = {};
        config.clear();
        metaDevices.clear();
    }

    void SetUp() override {
       // prepare mockExeNetwork
       mockIExeNet = std::make_shared<MockIExecutableNetworkInternal>();
       auto mockIPluginPtr = std::make_shared<MockIInferencePlugin>();
       ON_CALL(*mockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).WillByDefault(Return(mockIExeNet));
       mockPlugin = InferenceEngine::InferencePlugin{{}, mockIPluginPtr};
       mockExeNetwork = {{}, mockPlugin.LoadNetwork(CNNNetwork{}, {})};

       core  = std::shared_ptr<MockICore>(new MockICore());
       auto* origin_plugin = new MockMultiDeviceInferencePlugin();
       plugin  = std::shared_ptr<MockMultiDeviceInferencePlugin>(origin_plugin);
       function = ngraph::builder::subgraph::makeConvPoolRelu();
       cnnNet = InferenceEngine::CNNNetwork(function);
       // replace core with mock Icore
       plugin->SetCore(core);
       // mock execNetwork can work
       auto inferReqInternal = std::make_shared<MockIInferRequestInternal>();
       ON_CALL(*mockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(inferReqInternal));
       IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, optimalNum, 2);
       ON_CALL(*mockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)))).
           WillByDefault(Return(optimalNum));

       config.insert({CONFIG_KEY_INTERNAL(MULTI_WORK_MODE_AS_AUTO), InferenceEngine::PluginConfigParams::YES});
    }
    void SetUpTwoDevice() {
       config.insert({InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                      CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU});
       DeviceInformation GPU = {CommonTestUtils::DEVICE_GPU, {}, 2, ""};
       DeviceInformation CPU = {CommonTestUtils::DEVICE_CPU, {}, 2, ""};
       metaDevices.push_back(GPU);
       metaDevices.push_back(CPU);
       ON_CALL(*plugin, ParseMetaDevices(_, _)).WillByDefault(Return(metaDevices));

       ON_CALL(*plugin, SelectDevice(Property(&std::vector<DeviceInformation>::size, Eq(2)), _)).WillByDefault(Return(GPU));
       ON_CALL(*plugin, SelectDevice(Property(&std::vector<DeviceInformation>::size, Eq(1)), _)).WillByDefault(Return(CPU));
    }

    void SetUpTwoDeviceCase() {
       config.insert({InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                      CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU});
       DeviceInformation GPU = {CommonTestUtils::DEVICE_GPU, {}, 2, ""};
       DeviceInformation CPU = {CommonTestUtils::DEVICE_CPU, {}, 2, ""};
       metaDevices.push_back(GPU);
       metaDevices.push_back(CPU);
       ON_CALL(*plugin, ParseMetaDevices(_, _)).WillByDefault(Return(metaDevices));

       ON_CALL(*plugin, SelectDevice(Property(&std::vector<DeviceInformation>::size, Eq(2)), _)).WillByDefault(Return(GPU));
       ON_CALL(*plugin, SelectDevice(Property(&std::vector<DeviceInformation>::size, Eq(1)), _)).WillByDefault(Return(CPU));
    }

    void SetUpThreeDeviceCase() {
        config.insert({InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                      CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU +
                      std::string(",") + CommonTestUtils::DEVICE_MYRIAD});
        DeviceInformation GPU = {CommonTestUtils::DEVICE_GPU, {}, 2, ""};
        DeviceInformation CPU = {CommonTestUtils::DEVICE_CPU, {}, 2, ""};
        DeviceInformation MYRIAD = {CommonTestUtils::DEVICE_MYRIAD, {}, 2, ""};
        metaDevices.push_back(GPU);
        metaDevices.push_back(CPU);
        metaDevices.push_back(MYRIAD);
        ON_CALL(*plugin, ParseMetaDevices(_, _)).WillByDefault(Return(metaDevices));

        ON_CALL(*plugin, SelectDevice(Property(&std::vector<DeviceInformation>::size, Eq(3)), _)).WillByDefault(Return(GPU));
        ON_CALL(*plugin, SelectDevice(Property(&std::vector<DeviceInformation>::size, Eq(2)), _)).WillByDefault(Return(MYRIAD));
        ON_CALL(*plugin, SelectDevice(Property(&std::vector<DeviceInformation>::size, Eq(1)), _)).WillByDefault(Return(CPU));
    }
};

TEST_F(AutoLoadFailedTest, canContinueIfCpuFailedInTwoDevice) {
    // mock load GPU Success
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_GPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));

    // mock load CPU failed
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));

    SetUpTwoDevice();
    EXPECT_CALL(*plugin, SelectDevice(_, _)).Times(1);
    EXPECT_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(_),
                ::testing::Matcher<const Config&>(_))).Times(2);
    ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(cnnNet, config));
}

TEST_F(AutoLoadFailedTest, canContinueIfGpuFailedINTwoDevice) {
    // mock load GPU failed
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));

    // mock load CPU Success
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_GPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));

    SetUpTwoDevice();

    EXPECT_CALL(*plugin, SelectDevice(_, _)).Times(2);
    EXPECT_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(_),
                ::testing::Matcher<const Config&>(_))).Times(2);
    ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(cnnNet, config));
}

TEST_F(AutoLoadFailedTest, throwExceptionifCpuANDGpuFailed) {
    // mock GPU LOad failed
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_GPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));
    // mock CPU LOad failed
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));

    SetUpTwoDevice();

    EXPECT_CALL(*plugin, SelectDevice(_, _)).Times(2);
    EXPECT_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(_),
                ::testing::Matcher<const Config&>(_))).Times(2);
    ASSERT_THROW(plugin->LoadExeNetworkImpl(cnnNet, config), InferenceEngine::Exception);
}

TEST_F(AutoLoadFailedTest, throwExceptionifThreeDeviceAllFailed) {
    // mock GPU LOad failed
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_GPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));
    // mock CPU LOad failed
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));
    //mock MYRIAD LOad failed
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_MYRIAD)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));
    SetUpThreeDeviceCase();

    EXPECT_CALL(*plugin, SelectDevice(_, _)).Times(3);
    EXPECT_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(_),
                ::testing::Matcher<const Config&>(_))).Times(3);
    ASSERT_THROW(plugin->LoadExeNetworkImpl(cnnNet, config), InferenceEngine::Exception);
}

TEST_F(AutoLoadFailedTest, ContinueIfOnlyGPUfailInThreeDevice) {
    // mock GPU LOad failed
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_GPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));
    // mock CPU LOad Success
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));
    //mock MYRIAD LOad Success
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_MYRIAD)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));

    SetUpThreeDeviceCase();

    EXPECT_CALL(*plugin, SelectDevice(_, _)).Times(2);
    EXPECT_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(_),
                ::testing::Matcher<const Config&>(_))).Times(3);
    ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(cnnNet, config));
}

TEST_F(AutoLoadFailedTest, ContinueIfOnlyCPUfailInThreeDevice) {
    // mock GPU LOad success
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_GPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));
    // mock CPU LOad failed
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));
    //mock MYRIAD LOad success
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_MYRIAD)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));

    SetUpThreeDeviceCase();

    EXPECT_CALL(*plugin, SelectDevice(_, _)).Times(1);
    EXPECT_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(_),
                ::testing::Matcher<const Config&>(_))).Times(2);

    ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(cnnNet, config));
}

TEST_F(AutoLoadFailedTest, ContinueIfOnlyMYRIADfailInThreeDevice) {
    // mock GPU LOad success
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_GPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));
    // mock CPU LOad success
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));
    //mock MYRIAD LOad failed
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_MYRIAD)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));

    SetUpThreeDeviceCase();

    EXPECT_CALL(*plugin, SelectDevice(_, _)).Times(1);
    EXPECT_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(_),
                ::testing::Matcher<const Config&>(_))).Times(2);

    ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(cnnNet, config));
}

TEST_F(AutoLoadFailedTest, ContinueIfOnlyGPUSuccessInThreeDevice) {
    // mock GPU LOad success
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_GPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));
    // mock CPU LOad failed
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));
    //mock MYRIAD LOad failed
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_MYRIAD)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));

    SetUpThreeDeviceCase();

    EXPECT_CALL(*plugin, SelectDevice(_, _)).Times(1);
    EXPECT_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(_),
                ::testing::Matcher<const Config&>(_))).Times(2);

    ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(cnnNet, config));
}

TEST_F(AutoLoadFailedTest, ContinueIfOnlyMYRIADSuccessInThreeDevice) {
    // mock GPU LOad failed
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_GPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));
    // mock CPU LOad failed
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));
    //mock MYRIAD LOad Success
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_MYRIAD)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));

    SetUpThreeDeviceCase();

    EXPECT_CALL(*plugin, SelectDevice(_, _)).Times(2);
    EXPECT_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(_),
                ::testing::Matcher<const Config&>(_))).Times(3);

    ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(cnnNet, config));
}

TEST_F(AutoLoadFailedTest, ContinueIfOnlyCPUSuccessInThreeDevice) {
    // mock GPU LOad failed
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_GPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));
    // mock CPU LOad Success
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));
    //mock MYRIAD LOad failed
    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_MYRIAD)),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));

    SetUpThreeDeviceCase();

    EXPECT_CALL(*plugin, SelectDevice(_, _)).Times(3);
    EXPECT_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(_),
                ::testing::Matcher<const Config&>(_))).Times(3);

    ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(cnnNet, config));
}

