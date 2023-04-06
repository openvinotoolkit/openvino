// Copyright (C) 2018-2023 Intel Corporation
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
#include "plugin/mock_auto_device_plugin.hpp"
#include "mock_common.hpp"
#include <thread>

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
using ::testing::AtLeast;
using ::testing::AnyNumber;
using ::testing::InvokeWithoutArgs;
using ::testing::NiceMock;
using Config = std::map<std::string, std::string>;
using namespace MockMultiDevice;

using ConfigParams = std::tuple<
        bool,                 // cpu load success
        bool                  // hw device load success
        >;
class AutoReleaseHelperTest : public ::testing::TestWithParam<ConfigParams> {
public:
    std::shared_ptr<ngraph::Function>               function;
    InferenceEngine::CNNNetwork                     cnnNet;
    std::shared_ptr<NiceMock<MockICore>>                      core;
    std::shared_ptr<NiceMock<MockMultiDeviceInferencePlugin>> plugin;

    //mock exeNetwork helper
    ov::SoPtr<IExecutableNetworkInternal>  mockExeNetwork;
    //mock exeNetwork actual
    ov::SoPtr<IExecutableNetworkInternal>  mockExeNetworkActual;
    // config for Auto device
    std::map<std::string, std::string>              config;
    std::vector<DeviceInformation>                  metaDevices;
    std::shared_ptr<NiceMock<MockIInferRequestInternal>>     inferReqInternal;
    std::shared_ptr<NiceMock<MockIInferRequestInternal>>     inferReqInternalActual;
    size_t optimalNum;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        bool cpuSuccess;
        bool accSuccess;
        std::tie(cpuSuccess, accSuccess) = obj.param;
        std::ostringstream result;
         if (!cpuSuccess) {
            result << "cpuLoadFailure_";
        } else {
            result << "cpuLoadSuccess_";
        }
        if (!accSuccess) {
            result << "accelerateorLoadFailure";
        } else {
            result << "accelerateorLoadSuccess";
        }
        return result.str();
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
        //mockIExeNet.reset();
        mockExeNetwork = {};
        mockExeNetworkActual = {};
        config.clear();
        metaDevices.clear();
        inferReqInternal.reset();
        inferReqInternalActual.reset();
    }

    void SetUp() override {
       // prepare mockExeNetwork
       auto mockIExeNet = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
       mockExeNetwork = {mockIExeNet, {}};

       auto mockIExeNetActual = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
       mockExeNetworkActual = {mockIExeNetActual, {}};

       // prepare mockicore and cnnNetwork for loading
       core = std::make_shared<NiceMock<MockICore>>();
       NiceMock<MockMultiDeviceInferencePlugin>* mock_multi = new NiceMock<MockMultiDeviceInferencePlugin>();
       plugin.reset(mock_multi);
       function = ngraph::builder::subgraph::makeConvPoolRelu();
       cnnNet = InferenceEngine::CNNNetwork(function);
       // replace core with mock Icore
       plugin->SetCore(core);
       // mock execNetwork can work
       inferReqInternal = std::make_shared<NiceMock<MockIInferRequestInternal>>();
       ON_CALL(*mockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(inferReqInternal));
       IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, optimalNum, 1);
       ON_CALL(*mockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(Return(optimalNum));
       inferReqInternalActual = std::make_shared<NiceMock<MockIInferRequestInternal>>();
       ON_CALL(*mockIExeNetActual.get(), CreateInferRequest()).WillByDefault(Return(inferReqInternalActual));
       ON_CALL(*mockIExeNetActual.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(Return(optimalNum));
       IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, supportConfigs, {});
       ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _))
           .WillByDefault(Return(supportConfigs));
       ON_CALL(*core, GetConfig(_, StrEq(ov::compilation_num_threads.name())))
           .WillByDefault(Return(12));
    }
};

TEST_P(AutoReleaseHelperTest, releaseResource) {
    // get Parameter
    bool cpuSuccess;
    bool accSuccess;
    std::tie(cpuSuccess, accSuccess) = this->GetParam();
    size_t decreaseCount = 0;
    // test auto plugin
    plugin->SetName("AUTO");
    const std::string strDevices = CommonTestUtils::DEVICE_GPU + std::string(",") +
        CommonTestUtils::DEVICE_CPU;

    if (accSuccess) {
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_GPU)),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(InvokeWithoutArgs([this]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        return mockExeNetworkActual; }));
    } else {
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_GPU)),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(InvokeWithoutArgs([this]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        IE_THROW() << "";
                        return mockExeNetworkActual; }));
    }
    if (cpuSuccess) {
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));
            if (accSuccess)
                decreaseCount++;
    } else {
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));
    }
    metaDevices = {{CommonTestUtils::DEVICE_CPU, {}, -1}, {CommonTestUtils::DEVICE_GPU, {}, -1}};
    DeviceInformation devInfo;
    ON_CALL(*plugin, ParseMetaDevices(_, _)).WillByDefault(Return(metaDevices));
    ON_CALL(*plugin, GetValidDevice)
        .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
            std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
            return devices;
        });
    ON_CALL(*plugin, SelectDevice(Property(&std::vector<DeviceInformation>::size, Eq(2)), _, _))
            .WillByDefault(Return(metaDevices[1]));
    ON_CALL(*plugin, SelectDevice(Property(&std::vector<DeviceInformation>::size, Eq(1)), _, _))
            .WillByDefault(Return(metaDevices[0]));
    config.insert({InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
                  CommonTestUtils::DEVICE_CPU + std::string(",") + CommonTestUtils::DEVICE_GPU});
    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> exeNetwork;
    if (cpuSuccess || accSuccess) {
        ASSERT_NO_THROW(exeNetwork = plugin->LoadExeNetworkImpl(cnnNet, config));
        if (!cpuSuccess)
            EXPECT_EQ(exeNetwork->GetMetric(ov::execution_devices.name()).as<std::string>(), CommonTestUtils::DEVICE_GPU);
        else
            EXPECT_EQ(exeNetwork->GetMetric(ov::execution_devices.name()).as<std::string>(), "(CPU)");
    } else {
        ASSERT_THROW(exeNetwork = plugin->LoadExeNetworkImpl(cnnNet, config), InferenceEngine::Exception);
    }
    auto sharedcount = mockExeNetwork._ptr.use_count();
    auto requestsharedcount = inferReqInternal.use_count();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    EXPECT_EQ(mockExeNetwork._ptr.use_count(), sharedcount - decreaseCount);
    EXPECT_EQ(inferReqInternal.use_count(), requestsharedcount - decreaseCount);
    if (cpuSuccess || accSuccess) {
        if (accSuccess)
            EXPECT_EQ(exeNetwork->GetMetric(ov::execution_devices.name()).as<std::string>(), CommonTestUtils::DEVICE_GPU);
        else
            EXPECT_EQ(exeNetwork->GetMetric(ov::execution_devices.name()).as<std::string>(), CommonTestUtils::DEVICE_CPU);
    }
}

//
const std::vector<ConfigParams> testConfigs = {ConfigParams {true, true},
                                               ConfigParams {true, false},
                                               ConfigParams {false, true},
                                               ConfigParams {false, false}
                                              };

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, AutoReleaseHelperTest,
                ::testing::ValuesIn(testConfigs),
            AutoReleaseHelperTest::getTestCaseName);
