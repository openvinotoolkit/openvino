// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_metric_helpers.hpp>
#include <common_test_utils/test_constants.hpp>
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
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
using ::testing::ContainsRegex;
using Config = std::map<std::string, std::string>;
using namespace MockMultiDevice;

using DeviceParams = std::tuple<std::string, bool>;

enum MODEL {
    GENERAL = 0,
    LATENCY = 1,
    THROUGHPUT = 2,
};

using ConfigParams = std::tuple<
        bool,                        // if can continue to run
        bool,                        // if select throw exception
        MODEL,                       // config model general, latency, throughput
        std::vector<DeviceParams>,   // {device, loadSuccess}
        unsigned int,                // select count
        unsigned int,                // load count
        unsigned int                 // load device success count
        >;
class AutoLoadFailedTest : public ::testing::TestWithParam<ConfigParams> {
public:
    std::shared_ptr<ngraph::Function>               function;
    InferenceEngine::CNNNetwork                     cnnNet;
    std::shared_ptr<MockICore>                      core;
    std::shared_ptr<MockMultiDeviceInferencePlugin> plugin;

    //mock exeNetwork
    std::shared_ptr<MockIExecutableNetworkInternal> mockIExeNet;
    ov::SoPtr<IExecutableNetworkInternal>  mockExeNetwork;
    MockIInferencePlugin*                           mockIPlugin;
    std::shared_ptr<InferenceEngine::IInferencePlugin> mockPlugin;
    // config for Auto device
    std::map<std::string, std::string>              config;
    std::vector<DeviceInformation>                  metaDevices;
    std::shared_ptr<MockIInferRequestInternal>     inferReqInternal;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        unsigned int selectCount;
        unsigned int loadCount;
        unsigned int loadSuccessCount;
        std::vector<std::tuple<std::string, bool>> deviceConfigs;
        bool continueRun;
        bool thrExcWheSelect;
        MODEL configModel;
        std::tie(continueRun, thrExcWheSelect, configModel, deviceConfigs,
                 selectCount, loadCount, loadSuccessCount) = obj.param;
        std::ostringstream result;
        for (auto& item : deviceConfigs) {
            if (std::get<1>(item)) {
                result << std::get<0>(item) << "_success_";
            } else {
                result << std::get<0>(item) << "_failed_";
            }
        }
        if (thrExcWheSelect) {
            result << "select_failed_";
        } else {
            result << "select_success_";
        }

        switch (configModel) {
            case GENERAL:
                result << "GENERAL";
                break;
            case LATENCY:
                result << "LATENCY";
                break;
            case THROUGHPUT:
                result << "THROUGHPUT";
                break;
            default:
                LOG_ERROR("should not come here");
                break;
        }

        result << "select_" << selectCount << "_loadCount_"
               << loadCount << "_loadSuccessCount_" << loadSuccessCount;
        return result.str();
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
        mockIExeNet.reset();
        mockExeNetwork = {};
        mockPlugin = {};
        config.clear();
        metaDevices.clear();
        inferReqInternal.reset();
    }

    void SetUp() override {
       // prepare mockExeNetwork
       mockIExeNet = std::make_shared<MockIExecutableNetworkInternal>();
       auto mockIPluginPtr = std::make_shared<MockIInferencePlugin>();
       ON_CALL(*mockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).WillByDefault(Return(mockIExeNet));
       mockPlugin = mockIPluginPtr;
       // remove annoying ON CALL message
       EXPECT_CALL(*mockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).Times(1);
       mockExeNetwork = ov::SoPtr<InferenceEngine::IExecutableNetworkInternal>(mockPlugin->LoadNetwork(CNNNetwork{}, {}), {});

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
       ON_CALL(*mockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(inferReqInternal));
       IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, optimalNum, 2);
       ON_CALL(*mockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(Return(optimalNum));
       IE_SET_METRIC(OPTIMAL_BATCH_SIZE, optimalBatchSize, 8);
       ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(OPTIMAL_BATCH_SIZE)), _))
           .WillByDefault(Return(optimalBatchSize));
       EXPECT_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(OPTIMAL_BATCH_SIZE)), _)).Times(AnyNumber());
       IE_SET_METRIC(RANGE_FOR_STREAMS, rangeStreamsSize, {1u, 2u});
       ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(RANGE_FOR_STREAMS)), _))
           .WillByDefault(Return(rangeStreamsSize));
       EXPECT_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(RANGE_FOR_STREAMS)), _)).Times(AnyNumber());
       IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, supportConfigs, {});
       ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _))
           .WillByDefault(Return(supportConfigs));
       EXPECT_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _)).Times(AnyNumber());
       ON_CALL(*core, GetConfig(_, StrEq(ov::compilation_num_threads.name())))
           .WillByDefault(Return(12));
    }
};

TEST_P(AutoLoadFailedTest, LoadCNNetWork) {
    // get Parameter
    unsigned int selectCount;
    unsigned int loadCount;
    unsigned int loadSuccessCount;
    std::vector<std::tuple<std::string, bool>> deviceConfigs;
    bool continueRun;
    bool thrExcWheSelect;
    MODEL configModel;
    std::tie(continueRun, thrExcWheSelect, configModel, deviceConfigs, selectCount,
             loadCount, loadSuccessCount) = this->GetParam();

    // test auto plugin
    plugin->SetName("AUTO");
    std::string devicesStr = "";
    int selDevsSize = deviceConfigs.size();
    for (auto iter = deviceConfigs.begin(); iter != deviceConfigs.end(); selDevsSize--) {
        std::string deviceName = std::get<0>(*iter);
        bool loadSuccess = std::get<1>(*iter);
        // accoding to device loading config, set if the loading will successful or throw exception.
        if (loadSuccess) {
            ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                        ::testing::Matcher<const std::string&>(StrEq(deviceName)),
                        ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));
        } else {
            ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                        ::testing::Matcher<const std::string&>(StrEq(deviceName)),
                        ::testing::Matcher<const Config&>(_)))
                .WillByDefault(Throw(InferenceEngine::GeneralError{""}));
        }
        DeviceInformation devInfo;
        switch (configModel) {
            case GENERAL:
                devInfo = {deviceName, {}, 2, ""};
                break;
            case LATENCY:
                devInfo = {deviceName, {{CONFIG_KEY(PERFORMANCE_HINT),
                    InferenceEngine::PluginConfigParams::LATENCY}, {CONFIG_KEY(ALLOW_AUTO_BATCHING), "YES"}, {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "1000"}},
                    2, ""};
                break;
            case THROUGHPUT:
                devInfo = {deviceName, {{CONFIG_KEY(PERFORMANCE_HINT),
                    InferenceEngine::PluginConfigParams::THROUGHPUT}}, 2, ""};
                break;
            default:
                LOG_ERROR("should not come here");
                break;
        }

        metaDevices.push_back(std::move(devInfo));
        // set the return value of SelectDevice
        // for example if there are three device, if will return GPU on the first call, and then MYRIAD
        // at last CPU
        ON_CALL(*plugin, SelectDevice(Property(&std::vector<DeviceInformation>::size, Eq(selDevsSize)), _, _))
            .WillByDefault(Return(metaDevices[deviceConfigs.size() - selDevsSize]));
        devicesStr += deviceName;
        devicesStr += ((++iter) == deviceConfigs.end()) ? "" : ",";
    }
    ON_CALL(*plugin, ParseMetaDevices(_, _)).WillByDefault(Return(metaDevices));
    ON_CALL(*plugin, GetValidDevice)
        .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
            std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
            return devices;
        });
    config.insert({InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , devicesStr});
    // if set this parameter true, the second selecting call will thrown exception,
    // if there is only one device, it will thrown exception at the first call
    if (thrExcWheSelect) {
        selDevsSize = deviceConfigs.size();
        if (selDevsSize > 1) {
            ON_CALL(*plugin, SelectDevice(Property(&std::vector<DeviceInformation>::size, Eq(selDevsSize - 1)), _, _))
                .WillByDefault(Throw(InferenceEngine::GeneralError{""}));
        } else {
            ON_CALL(*plugin, SelectDevice(Property(&std::vector<DeviceInformation>::size, Eq(1)), _, _))
                .WillByDefault(Throw(InferenceEngine::GeneralError{""}));
        }
    }

    EXPECT_CALL(*plugin, ParseMetaDevices(_, _)).Times(AtLeast(1));
    EXPECT_CALL(*plugin, SelectDevice(_, _, _)).Times(selectCount);
    EXPECT_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(_),
                ::testing::Matcher<const Config&>(_))).Times(loadCount);

    // if loadSuccess will get the optimalNum requset of per device, in this test is 2;
    EXPECT_CALL(*mockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
        .Times(loadSuccessCount);
    EXPECT_CALL(*inferReqInternal, SetCallback(_)).Times(loadSuccessCount * 2);
    EXPECT_CALL(*mockIExeNet.get(), CreateInferRequest()).Times(loadSuccessCount * 2);
    if (continueRun) {
        ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(cnnNet, config));
    } else {
        ASSERT_THROW(plugin->LoadExeNetworkImpl(cnnNet, config), InferenceEngine::Exception);
    }
}

// the test configure, for example
// ConfigParams {true, false,  GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, false},
//               DeviceParams {CommonTestUtils::DEVICE_KEEMBAY, true},
//                DeviceParams {CommonTestUtils::DEVICE_CPU, true}}, 2, 3, 2},
//
// every element for ConfigParams
// {continueRun, selectThrowException,  config model,  deviceLoadsuccessVector, selectCount, loadCount, loadSuccessCount}
// {       true,                false,       GENERAL,                 3 device,           2,         3,                2}
//
// there are three devices for loading
// CPU load for accelerator success, but GPU will load faild and then select MYRIAD and load again
// LoadExeNetworkImpl will not throw exception and can continue to run,
// it will select twice, first select GPU, second select MYRIAD
// it will load network three times(CPU, GPU, MYRIAD)
// the inference request num is loadSuccessCount * optimalNum, in this test case optimalNum is 2
// so inference request num is 4 (CPU 2, MYRIAD 2)
//
const std::vector<ConfigParams> testConfigs = {ConfigParams {true, false, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, true},
                                                        DeviceParams {CommonTestUtils::DEVICE_KEEMBAY, true},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, true}}, 1, 2, 2},
                                               ConfigParams {true, false, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_KEEMBAY, true},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, true}}, 2, 3, 2},
                                               ConfigParams {true, false, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, true},
                                                        DeviceParams {CommonTestUtils::DEVICE_KEEMBAY, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, true}}, 1, 2, 2},
                                               ConfigParams {true, false, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, true},
                                                        DeviceParams {CommonTestUtils::DEVICE_KEEMBAY, true},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, false}}, 1, 2, 1},
                                               ConfigParams {true, false, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, true},
                                                        DeviceParams {CommonTestUtils::DEVICE_KEEMBAY, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, false}}, 1, 2, 1},
                                               ConfigParams {true, false, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_KEEMBAY, true},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, false}}, 2, 3, 1},
                                               ConfigParams {true, false, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_KEEMBAY, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, true}}, 3, 4, 2},
                                               ConfigParams {false, false, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_KEEMBAY, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, false}}, 3, 4, 0},
                                               ConfigParams {true, false, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, true},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, true}}, 1, 2, 2},
                                               ConfigParams {true, false, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, true}}, 2, 3, 2},
                                               ConfigParams {true, false, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, true},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, false}}, 1, 2, 1},
                                               ConfigParams {false, false, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, false}}, 2, 3, 0},
                                               ConfigParams {false, false, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, false}}, 1, 1, 0},
                                               ConfigParams {false, false, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_CPU, false}}, 1, 1, 0},
                                               ConfigParams {true, false, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, true}}, 1, 1, 1},
                                               ConfigParams {true, false, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_CPU, true}}, 1, 1, 1},
                                               ConfigParams {false, true, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, true}}, 1, 0, 0},
                                               ConfigParams {false, true, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_CPU, true}}, 1, 0, 0},
                                               ConfigParams {true, true, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_KEEMBAY, true},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, true}}, 2, 2, 1},
                                               ConfigParams {false, true, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_KEEMBAY, true},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, false}}, 2, 2, 0},
                                               ConfigParams {true, true, GENERAL, {DeviceParams {CommonTestUtils::DEVICE_GPU, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, true}}, 2, 2, 1},
                                               ConfigParams {true, false, LATENCY, {DeviceParams {CommonTestUtils::DEVICE_GPU, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_KEEMBAY, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, true}}, 3, 3, 1},
                                               ConfigParams {true, false, LATENCY, {DeviceParams {CommonTestUtils::DEVICE_GPU, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, true}}, 2, 2, 1},
                                               ConfigParams {true, false, THROUGHPUT, {DeviceParams {CommonTestUtils::DEVICE_GPU, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_KEEMBAY, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, true}}, 3, 4, 2},
                                               ConfigParams {true, false, THROUGHPUT, {DeviceParams {CommonTestUtils::DEVICE_GPU, false},
                                                        DeviceParams {CommonTestUtils::DEVICE_CPU, true}}, 2, 3, 2}
                                              };

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, AutoLoadFailedTest,
                ::testing::ValuesIn(testConfigs),
            AutoLoadFailedTest::getTestCaseName);

