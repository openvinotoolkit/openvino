// Copyright (C) 2018-2022 Intel Corporation
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
#include "cpp/ie_plugin.hpp"
#include <chrono>
#include <thread>
#include "mock_common.hpp"

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
using Config = std::map<std::string, std::string>;
using namespace MockMultiDevice;

using ConfigParams = std::tuple<
        bool,                        // if THROUGHPUT
        unsigned int,                // cpu OPTIMAL_NUMBER_OF_INFER_REQUESTS
        int,                         // cpu infer requet num of customer want
        bool,                        // if cpu sleep, cpu device will load slow
        unsigned int,                // gpu OPTIMAL_NUMBER_OF_INFER_REQUESTS
        int,                         // gpu infer requet num of customer want
        bool,                        // if gpu sleep, cpu device will load slow
        unsigned int                 // expect OPTIMAL_NUMBER_OF_INFER_REQUESTS
        >;
class ExecNetworkGetMetric : public ::testing::TestWithParam<ConfigParams> {
public:
    std::shared_ptr<ngraph::Function>               function;
    InferenceEngine::CNNNetwork                     cnnNet;
    std::shared_ptr<MockICore>                      core;
    std::shared_ptr<MockMultiDeviceInferencePlugin> plugin;

    //mock cpu exeNetwork
    std::shared_ptr<MockIExecutableNetworkInternal> cpuMockIExeNet;
    ov::runtime::SoPtr<IExecutableNetworkInternal>  cpuMockExeNetwork;
    MockIInferencePlugin*                           cpuMockIPlugin;
    InferenceEngine::InferencePlugin                cpuMockPlugin;
    //mock gpu exeNetwork
    std::shared_ptr<MockIExecutableNetworkInternal> gpuMockIExeNet;
    ov::runtime::SoPtr<IExecutableNetworkInternal>  gpuMockExeNetwork;
    MockIInferencePlugin*                           gpuMockIPlugin;
    InferenceEngine::InferencePlugin                gpuMockPlugin;
    // config for Auto device
    std::map<std::string, std::string>              config;
    std::vector<DeviceInformation>                  metaDevices;
    std::shared_ptr<MockIInferRequestInternal>      inferReqInternal;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        unsigned int cpuOptimalNum;
        int cpuCustomerNum;
        unsigned int gpuOptimalNum;
        int gpuCustomerNum;
        unsigned int expectOptimalNum;
        bool cpuSleep;
        bool gpuSleep;
        bool isThroughput;
        std::tie(isThroughput, cpuOptimalNum, cpuCustomerNum, cpuSleep,
                 gpuOptimalNum, gpuCustomerNum, gpuSleep, expectOptimalNum) = obj.param;
        std::ostringstream result;
        result << "cpuOptimalNum_" << cpuOptimalNum << "cpuCustomerNum_" << cpuCustomerNum;
        result << "gpuOptimalNum_" << gpuOptimalNum << "gpuCustomerNum_" << gpuCustomerNum;
        result << "expectOptimalNum_" << expectOptimalNum;
        if (isThroughput) {
            result << "_isThroughput" << "true";
        } else {
            result << "__isThroughput" << "false";
        }
        if (cpuSleep) {
            result << "_cpuSleep_" << "true";
        } else {
            result << "_cpuSleep_" << "false";
        }

        if (gpuSleep) {
            result << "_gpuSleep_" << "true";
        } else {
            result << "_gpuSleep_" << "false";
        }

        return result.str();
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
        cpuMockIExeNet.reset();
        cpuMockExeNetwork = {};
        cpuMockPlugin = {};
        gpuMockIExeNet.reset();
        gpuMockExeNetwork = {};
        gpuMockPlugin = {};
        config.clear();
        metaDevices.clear();
        inferReqInternal.reset();
    }

    void SetUp() override {
       // prepare cpuMockExeNetwork
       cpuMockIExeNet = std::make_shared<MockIExecutableNetworkInternal>();
       auto cpuMockIPluginPtr = std::make_shared<MockIInferencePlugin>();
       ON_CALL(*cpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).WillByDefault(Return(cpuMockIExeNet));
       cpuMockPlugin = InferenceEngine::InferencePlugin{cpuMockIPluginPtr, {}};
       // remove annoying ON CALL message
       EXPECT_CALL(*cpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).Times(1);
       cpuMockExeNetwork = cpuMockPlugin.LoadNetwork(CNNNetwork{}, {});

       // prepare gpuMockExeNetwork
       gpuMockIExeNet = std::make_shared<MockIExecutableNetworkInternal>();
       auto gpuMockIPluginPtr = std::make_shared<MockIInferencePlugin>();
       ON_CALL(*gpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).WillByDefault(Return(gpuMockIExeNet));
       gpuMockPlugin = InferenceEngine::InferencePlugin{gpuMockIPluginPtr, {}};
       // remove annoying ON CALL message
       EXPECT_CALL(*gpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).Times(1);
       gpuMockExeNetwork = gpuMockPlugin.LoadNetwork(CNNNetwork{}, {});

       // prepare mockicore and cnnNetwork for loading
       core  = std::shared_ptr<MockICore>(new MockICore());
       auto* origin_plugin = new MockMultiDeviceInferencePlugin();
       plugin  = std::shared_ptr<MockMultiDeviceInferencePlugin>(origin_plugin);
       function = ngraph::builder::subgraph::makeConvPoolRelu();
       cnnNet = InferenceEngine::CNNNetwork(function);
       // replace core with mock Icore
       plugin->SetCore(core);
       // mock execNetwork can work
       inferReqInternal = std::make_shared<MockIInferRequestInternal>();
       ON_CALL(*cpuMockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(inferReqInternal));
       ON_CALL(*gpuMockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(inferReqInternal));
       EXPECT_CALL(*inferReqInternal, SetCallback).Times(AtLeast(1));
       IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, supportConfigs, {});
       ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _))
           .WillByDefault(RETURN_MOCK_VALUE(supportConfigs));
       EXPECT_CALL(*core, GetMetric(_, _, _)).Times(AnyNumber());

       // test auto plugin
       config.insert({CONFIG_KEY_INTERNAL(MULTI_WORK_MODE_AS_AUTO), InferenceEngine::PluginConfigParams::YES});
       config.insert({InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
               CommonTestUtils::DEVICE_CPU + std::string(",") + CommonTestUtils::DEVICE_GPU});

       ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
                    std::cout << stream.str() << std::endl;
               });
    }
};

TEST_P(ExecNetworkGetMetric, OPTIMAL_NUMBER_OF_INFER_REQUESTS) {
    unsigned int cpuOptimalNum;
    int cpuCustomerNum;
    unsigned int gpuOptimalNum;
    int gpuCustomerNum;
    unsigned int expectOptimalNum;
    bool cpuSleep;
    bool gpuSleep;
    bool isThroughput;
    std::tie(isThroughput, cpuOptimalNum, cpuCustomerNum, cpuSleep,
             gpuOptimalNum, gpuCustomerNum, gpuSleep, expectOptimalNum) = this->GetParam();
    if (isThroughput) {
        metaDevices.push_back({CommonTestUtils::DEVICE_CPU, {{CONFIG_KEY(PERFORMANCE_HINT),
                    InferenceEngine::PluginConfigParams::THROUGHPUT}}, cpuCustomerNum, ""});
        metaDevices.push_back({CommonTestUtils::DEVICE_GPU, {{CONFIG_KEY(PERFORMANCE_HINT),
                    InferenceEngine::PluginConfigParams::THROUGHPUT}}, gpuCustomerNum, ""});
        IE_SET_METRIC(OPTIMAL_BATCH_SIZE, optimalBatchNum, 256);
        IE_SET_METRIC(RANGE_FOR_STREAMS, rangeOfStreams, std::make_tuple<unsigned int, unsigned int>(1, 2));
        ON_CALL(*core.get(), GetMetric(StrEq(CommonTestUtils::DEVICE_GPU), StrEq(METRIC_KEY(OPTIMAL_BATCH_SIZE)), _))
            .WillByDefault(RETURN_MOCK_VALUE(optimalBatchNum));
        ON_CALL(*core.get(), GetMetric(StrEq(CommonTestUtils::DEVICE_GPU), StrEq(METRIC_KEY(RANGE_FOR_STREAMS)), _))
            .WillByDefault(RETURN_MOCK_VALUE(rangeOfStreams));
    } else {
        metaDevices.push_back({CommonTestUtils::DEVICE_CPU, {}, cpuCustomerNum, ""});
        metaDevices.push_back({CommonTestUtils::DEVICE_GPU, {}, gpuCustomerNum, ""});
    }
    ON_CALL(*plugin, SelectDevice(_, _, _)).WillByDefault(Return(metaDevices[1]));
    ON_CALL(*plugin, ParseMetaDevices(_, _)).WillByDefault(Return(metaDevices));
    EXPECT_CALL(*plugin, ParseMetaDevices(_, _)).Times(1);
    EXPECT_CALL(*plugin, SelectDevice(_, _, _)).Times(1);

    if (cpuSleep) {
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(InvokeWithoutArgs([this]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                return cpuMockExeNetwork;
                }));
    } else {
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(Return(cpuMockExeNetwork));
    }

    if (gpuSleep) {
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_GPU)),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(InvokeWithoutArgs([this]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                return gpuMockExeNetwork;
                }));
    } else {
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_GPU)),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(Return(gpuMockExeNetwork));
    }

    ON_CALL(*cpuMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(RETURN_MOCK_VALUE(cpuOptimalNum));
    ON_CALL(*gpuMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(RETURN_MOCK_VALUE(gpuOptimalNum));

    EXPECT_CALL(*cpuMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .Times(AtLeast(1));

    EXPECT_CALL(*gpuMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .Times(AtLeast(1));

    EXPECT_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(CommonTestUtils::DEVICE_CPU),
                ::testing::Matcher<const Config&>(_))).Times(1);

    EXPECT_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(CommonTestUtils::DEVICE_GPU),
                ::testing::Matcher<const Config&>(_))).Times(1);

    if (cpuCustomerNum == -1) {
        EXPECT_CALL(*cpuMockIExeNet.get(), CreateInferRequest()).Times(cpuOptimalNum);
    } else {
        EXPECT_CALL(*cpuMockIExeNet.get(), CreateInferRequest()).Times(cpuCustomerNum);
    }

    if (gpuCustomerNum == -1) {
        EXPECT_CALL(*gpuMockIExeNet.get(), CreateInferRequest()).Times(gpuOptimalNum);
    } else {
        EXPECT_CALL(*gpuMockIExeNet.get(), CreateInferRequest()).Times(gpuCustomerNum);
    }

    auto AutoExecNetwork =  plugin->LoadExeNetworkImpl(cnnNet, config);
    auto result = AutoExecNetwork->GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
    EXPECT_EQ(result, expectOptimalNum);
}


// ConfigParams {bool, unsigned int, int, bool,
//               unsigned int, int, bool, unsigned int}
//
// every element for ConfigParams
// {is throughput mode, cpuOptimalNum, customer hope for cpu infer requset num, if cpu sleep when load,
//  gpuOptimalNum, customer hope for gpu infer requset num, if gpu sleep when load,
//  expectOptimalNum of Auto ExecNetwork}
//
const std::vector<ConfigParams> testConfigs = {
                                               ConfigParams {false, 1, -1, false, 2, -1, true, 8},
                                               ConfigParams {false, 1, -1, false, 10, -1, true, 8},
                                               ConfigParams {false, 12, -1, false, 2, -1, true, 12},
                                               ConfigParams {false, 12, -1, false, 10, -1, true, 12},
                                               ConfigParams {false, 1, -1, true, 2, -1, false, 8},
                                               ConfigParams {false, 1, -1, true, 10, -1, false, 10},
                                               ConfigParams {false, 6, -1, true, 2, -1, false, 8},
                                               ConfigParams {false, 6, -1, true, 10, -1, false, 10},
                                               ConfigParams {false, 6, 4, false, 2, 3, true, 8},
                                               ConfigParams {false, 6, 4, false, 10, 3, true, 8},
                                               ConfigParams {false, 1, 4, true, 2, 3, false, 8},
                                               ConfigParams {false, 1, 4, true, 10, 3, false, 10},
                                               ConfigParams {true, 1, 4, false, 10, 3, true, 512}
                                              };

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, ExecNetworkGetMetric,
                ::testing::ValuesIn(testConfigs),
            ExecNetworkGetMetric::getTestCaseName);
