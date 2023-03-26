// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_metric_helpers.hpp>
#include <common_test_utils/test_constants.hpp>
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "so_ptr.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include <ie_core.hpp>
#include <memory>
#include <multi-device/multi_device_config.hpp>
#include <ngraph_functions/subgraph_builders.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "plugin/mock_auto_device_plugin.hpp"
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
using ::testing::ContainsRegex;
using Config = std::map<std::string, std::string>;
using namespace MockMultiDevice;
class ExecNetworkGetMetricBase : public ::testing::Test {
public:
    std::shared_ptr<ngraph::Function>               function;
    InferenceEngine::CNNNetwork                     cnnNet;
    std::shared_ptr<MockICore>                      core;
    std::shared_ptr<MockMultiDeviceInferencePlugin> plugin;

    //mock cpu exeNetwork
    std::shared_ptr<MockIExecutableNetworkInternal> cpuMockIExeNet;
    ov::SoPtr<IExecutableNetworkInternal>  cpuMockExeNetwork;
    MockIInferencePlugin*                           cpuMockIPlugin;
    std::shared_ptr<InferenceEngine::IInferencePlugin> cpuMockPlugin;

    //mock actual exeNetwork
    std::shared_ptr<MockIExecutableNetworkInternal> actualMockIExeNet;
    ov::SoPtr<IExecutableNetworkInternal>  actualMockExeNetwork;
    MockIInferencePlugin*                           actualMockIPlugin;
    std::shared_ptr<InferenceEngine::IInferencePlugin> actualMockPlugin;

    // config for Auto device
    std::map<std::string, std::string>              config;
    std::vector<DeviceInformation>                  metaDevices;
    std::shared_ptr<MockIInferRequestInternal>      inferReqInternal;

public:
    void TearDown() override {
        core.reset();
        plugin.reset();
        cpuMockIExeNet.reset();
        cpuMockExeNetwork = {};
        cpuMockPlugin = {};
        actualMockIExeNet.reset();
        actualMockExeNetwork = {};
        actualMockPlugin = {};
        config.clear();
        metaDevices.clear();
        inferReqInternal.reset();
    }

    void SetUp() override {
       // prepare cpuMockExeNetwork
       cpuMockIExeNet = std::make_shared<MockIExecutableNetworkInternal>();
       auto cpuMockIPluginPtr = std::make_shared<MockIInferencePlugin>();
       ON_CALL(*cpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).WillByDefault(Return(cpuMockIExeNet));
       cpuMockPlugin = cpuMockIPluginPtr;
       // remove annoying ON CALL message
       EXPECT_CALL(*cpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).Times(1);
       cpuMockExeNetwork = ov::SoPtr<InferenceEngine::IExecutableNetworkInternal>(cpuMockPlugin->LoadNetwork(CNNNetwork{}, {}), {});

       // prepare actualMockExeNetwork
       actualMockIExeNet = std::make_shared<MockIExecutableNetworkInternal>();
       auto actualMockIPluginPtr = std::make_shared<MockIInferencePlugin>();
       ON_CALL(*actualMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).WillByDefault(Return(actualMockIExeNet));
       actualMockPlugin = actualMockIPluginPtr;
       // remove annoying ON CALL message
       EXPECT_CALL(*actualMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).Times(1);
       actualMockExeNetwork = ov::SoPtr<InferenceEngine::IExecutableNetworkInternal>(actualMockPlugin->LoadNetwork(CNNNetwork{}, {}), {});

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
       ON_CALL(*actualMockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(inferReqInternal));
       //EXPECT_CALL(*inferReqInternal, SetCallback).Times(AtLeast(1));
       IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, supportConfigs, {});
       ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _))
           .WillByDefault(RETURN_MOCK_VALUE(supportConfigs));
       EXPECT_CALL(*core, GetMetric(_, _, _)).Times(AnyNumber());

       // test auto plugin
       plugin->SetName("AUTO");
    }
};

using ConfigParams = std::tuple<bool,          // if THROUGHPUT
                                unsigned int,  // cpu OPTIMAL_NUMBER_OF_INFER_REQUESTS
                                int,           // cpu infer requet num of customer want
                                bool,          // if cpu sleep, cpu device will load slow
                                unsigned int,  // Actual device OPTIMAL_NUMBER_OF_INFER_REQUESTS
                                int,           // Actual device infer requet num of customer want
                                bool,          // if Actual device sleep, cpu device will load slow
                                std::string,   // Actual Device Name
                                unsigned int,  // expect OPTIMAL_NUMBER_OF_INFER_REQUESTS
                                int            // Actual PERFORMANCE_HINT_NUM_REQUESTS
                                >;
class ExecNetworkGetMetricOptimalNumInferReq : public ExecNetworkGetMetricBase,
                                               public ::testing::WithParamInterface<ConfigParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        unsigned int cpuOptimalNum;
        int cpuCustomerNum;
        unsigned int actualOptimalNum;
        int actualCustomerNum;
        unsigned int expectOptimalNum;
        bool cpuSleep;
        bool actualSleep;
        bool isThroughput;
        int gpuPerfHintNum;
        std::string actualDeviceName;
        std::tie(isThroughput,
                 cpuOptimalNum,
                 cpuCustomerNum,
                 cpuSleep,
                 actualOptimalNum,
                 actualCustomerNum,
                 actualSleep,
                 actualDeviceName,
                 expectOptimalNum,
                 gpuPerfHintNum) = obj.param;
        std::ostringstream result;
        result << "cpuOptimalNum_" << cpuOptimalNum << "cpuCustomerNum_" << cpuCustomerNum;
        result << "actualOptimalNum_" << actualOptimalNum << "actualCustomerNum_" << actualCustomerNum;
        result << "expectOptimalNum_" << expectOptimalNum;
        if (isThroughput) {
            result << "_isThroughput"
                   << "true";
        } else {
            result << "__isThroughput"
                   << "false";
        }
        if (cpuSleep) {
            result << "_cpuSleep_"
                   << "true";
        } else {
            result << "_cpuSleep_"
                   << "false";
        }

        if (actualSleep) {
            result << "_actualSleep_"
                   << "true";
        } else {
            result << "_actualSleep_"
                   << "false";
        }
        result << "_actualDeviceName_" << actualDeviceName;
        result << "_gpuPerfHintNum_" << gpuPerfHintNum;
        return result.str();
    }
};

using modelPrioPerfHintTestParams = std::tuple<bool,          // is New API
                                               bool,          // if Actual device sleep, cpu device will load slow
                                               std::string,   // Actual Device Name
                                               std::string,   // performance mode
                                               IE::Parameter  // model Priority
                                               >;

class ExecNetworkGetMetricOtherTest : public ExecNetworkGetMetricBase,
                                               public ::testing::WithParamInterface<modelPrioPerfHintTestParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<modelPrioPerfHintTestParams> obj) {
        bool isNewAPI;
        bool actualSleep;
        std::string actualDeviceName;
        std::string performanceMode;
        IE::Parameter modelPriority;
        std::tie(isNewAPI,
                 actualSleep,
                 actualDeviceName,
                 performanceMode,
                 modelPriority) = obj.param;
        std::ostringstream result;
        if (isNewAPI) {
            result << "_isNewAPI_"
                   << "true";
        } else {
            result << "_isNewAPI_"
                   << "false";
        }
        if (actualSleep) {
            result << "_actualSleep_"
                   << "true";
        } else {
            result << "_actualSleep_"
                   << "false";
        }
        result << "_actualDeviceName_" << actualDeviceName;
        result << "_performanceMode_" << performanceMode;
        result << "_modelPriority" << modelPriority.as<std::string>();
        return result.str();
    }
};

TEST_P(ExecNetworkGetMetricOptimalNumInferReq, OPTIMAL_NUMBER_OF_INFER_REQUESTS) {
    unsigned int cpuOptimalNum;
    int cpuCustomerNum;
    unsigned int actualOptimalNum;
    int actualCustomerNum;
    unsigned int expectOptimalNum;
    bool cpuSleep;
    bool actualSleep;
    bool isThroughput;
    int gpuPerfHintNum;
    std::string actualDeviceName;
    std::tie(isThroughput, cpuOptimalNum, cpuCustomerNum, cpuSleep, actualOptimalNum,
                actualCustomerNum, actualSleep, actualDeviceName, expectOptimalNum, gpuPerfHintNum) = this->GetParam();
    config.insert({InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
            CommonTestUtils::DEVICE_CPU + std::string(",") + actualDeviceName});
    if (isThroughput) {
        metaDevices.push_back({CommonTestUtils::DEVICE_CPU, {{CONFIG_KEY(PERFORMANCE_HINT),
                    InferenceEngine::PluginConfigParams::THROUGHPUT}}, cpuCustomerNum, ""});
        metaDevices.push_back({actualDeviceName, {{CONFIG_KEY(PERFORMANCE_HINT),
                    InferenceEngine::PluginConfigParams::THROUGHPUT}}, actualCustomerNum, ""});
        // enable autoBatch
        IE_SET_METRIC(OPTIMAL_BATCH_SIZE, gpuOptimalBatchNum, 8);
        IE_SET_METRIC(OPTIMAL_BATCH_SIZE, keembayOptimalBatchNum, 1);
        IE_SET_METRIC(RANGE_FOR_STREAMS, rangeOfStreams, std::make_tuple<unsigned int, unsigned int>(1, 3));
        ON_CALL(*core.get(), GetMetric(StrEq(CommonTestUtils::DEVICE_GPU), StrEq(METRIC_KEY(OPTIMAL_BATCH_SIZE)), _))
            .WillByDefault(RETURN_MOCK_VALUE(gpuOptimalBatchNum));
        ON_CALL(*core.get(), GetMetric(StrEq(CommonTestUtils::DEVICE_KEEMBAY), StrEq(METRIC_KEY(OPTIMAL_BATCH_SIZE)), _))
            .WillByDefault(RETURN_MOCK_VALUE(keembayOptimalBatchNum));
        ON_CALL(*core.get(), GetMetric(_, StrEq(METRIC_KEY(RANGE_FOR_STREAMS)), _))
            .WillByDefault(RETURN_MOCK_VALUE(rangeOfStreams));
        ON_CALL(*core.get(), GetConfig(_, StrEq(CONFIG_KEY(PERFORMANCE_HINT))))
            .WillByDefault(Return(CONFIG_VALUE(THROUGHPUT)));
        EXPECT_CALL(*core.get(), GetConfig(_, StrEq(CONFIG_KEY(PERFORMANCE_HINT)))).Times(AnyNumber());
        ON_CALL(*core.get(), GetConfig(_, StrEq(CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS))))
            .WillByDefault(Return(std::to_string(gpuPerfHintNum)));
        EXPECT_CALL(*core.get(), GetConfig(_, StrEq(CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS)))).Times(AnyNumber());
        ON_CALL(*core, GetConfig(_, StrEq(ov::compilation_num_threads.name())))
           .WillByDefault(Return(8));
        EXPECT_CALL(*core.get(), GetConfig(_, StrEq(ov::compilation_num_threads.name()))).Times(AnyNumber());
    } else {
        metaDevices.push_back({CommonTestUtils::DEVICE_CPU, {}, cpuCustomerNum, ""});
        metaDevices.push_back({actualDeviceName, {}, actualCustomerNum, ""});
        ON_CALL(*core, GetConfig(_, StrEq(ov::compilation_num_threads.name()))).WillByDefault(Return(8));
    }
    ON_CALL(*plugin, SelectDevice(_, _, _)).WillByDefault(Return(metaDevices[1]));
    ON_CALL(*plugin, ParseMetaDevices(_, _)).WillByDefault(Return(metaDevices));
    ON_CALL(*plugin, GetValidDevice)
        .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
            std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
            return devices;
        });
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

    if (actualSleep) {
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(StrEq(actualDeviceName)),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(InvokeWithoutArgs([this]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                return actualMockExeNetwork;
                }));
    } else {
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(StrEq(actualDeviceName)),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(Return(actualMockExeNetwork));
    }

    // ON_CALL(*core, GetConfig(::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_GPU)),
    //             ::testing::Matcher<const std::string&>(StrEq(CONFIG_KEY(GPU_THROUGHPUT_STREAMS))))).WillByDefault(Return("2"));

    ON_CALL(*cpuMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(RETURN_MOCK_VALUE(cpuOptimalNum));
    ON_CALL(*actualMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(RETURN_MOCK_VALUE(actualOptimalNum));

    EXPECT_CALL(*cpuMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .Times(AtLeast(1));

    EXPECT_CALL(*actualMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .Times(AtLeast(1));

    EXPECT_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(CommonTestUtils::DEVICE_CPU),
                ::testing::Matcher<const Config&>(_))).Times(1);

    EXPECT_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(actualDeviceName),
                ::testing::Matcher<const Config&>(_))).Times(1);

    if (cpuCustomerNum == -1) {
        EXPECT_CALL(*cpuMockIExeNet.get(), CreateInferRequest()).Times(cpuOptimalNum);
    } else {
        EXPECT_CALL(*cpuMockIExeNet.get(), CreateInferRequest()).Times(cpuCustomerNum);
    }

    if (actualCustomerNum == -1) {
        EXPECT_CALL(*actualMockIExeNet.get(), CreateInferRequest()).Times(actualOptimalNum);
    } else {
        EXPECT_CALL(*actualMockIExeNet.get(), CreateInferRequest()).Times(actualCustomerNum);
    }

    auto AutoExecNetwork =  plugin->LoadExeNetworkImpl(cnnNet, config);
    auto result = AutoExecNetwork->GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
    EXPECT_EQ(result, expectOptimalNum);
}

// ConfigParams {bool, unsigned int, int, bool,
//               unsigned int, int, bool, std::string, unsigned int}
//
// every element for ConfigParams
// {is throughput mode, cpuOptimalNum, customer hope for cpu infer requset num, if cpu sleep when load,
//  actualOptimalNum, customer hope for actual infer requset num, if actual sleep when load, actual device Name
//  expectOptimalNum of Auto ExecNetwork}
//
const std::vector<ConfigParams> testConfigs = {
                                               ConfigParams {false, 3, -1, false, 2, -1, true, CommonTestUtils::DEVICE_GPU,  1, 0},
                                               ConfigParams {true,  3, -1, false, 2, -1, true, CommonTestUtils::DEVICE_GPU,  48, 0},
                                               ConfigParams {false, 3, -1, true, 2, -1, false, CommonTestUtils::DEVICE_GPU,  2, 0},
                                               ConfigParams {true,  3, -1, true, 2, -1, false, CommonTestUtils::DEVICE_GPU,  2, 0},
                                               ConfigParams {false, 3, 5, false, 2, 5, true, CommonTestUtils::DEVICE_GPU,  1, 0},
                                               ConfigParams {true,  3, 5, false, 2, 5, true, CommonTestUtils::DEVICE_GPU,  48, 0},
                                               ConfigParams {false, 3, 5, true, 2, 5, false, CommonTestUtils::DEVICE_GPU,  2, 0},
                                               ConfigParams {true,  3, 5, true, 2, 5, false, CommonTestUtils::DEVICE_GPU,  2, 0},
                                               ConfigParams {true,  3, 5, false, 2, 5, true, CommonTestUtils::DEVICE_GPU,  48, 48},
                                               ConfigParams {true,  3, 5, false, 2, 5, true, CommonTestUtils::DEVICE_GPU,  8, 6},
                                               ConfigParams {false, 3, -1, false, 2, -1, true, CommonTestUtils::DEVICE_KEEMBAY,  1, 0},
                                               ConfigParams {true,  3, -1, false, 2, -1, true, CommonTestUtils::DEVICE_KEEMBAY,  8, 0},
                                               ConfigParams {false, 3, -1, true, 2, -1, false, CommonTestUtils::DEVICE_KEEMBAY,  2, 0},
                                               ConfigParams {true,  3, -1, true, 2, -1, false, CommonTestUtils::DEVICE_KEEMBAY,  2, 0},
                                               ConfigParams {false, 3, 5, false, 2, 5, true, CommonTestUtils::DEVICE_KEEMBAY,  1, 0},
                                               ConfigParams {true,  3, 5, false, 2, 5, true, CommonTestUtils::DEVICE_KEEMBAY,  8, 0},
                                               ConfigParams {false, 3, 5, true, 2, 5, false, CommonTestUtils::DEVICE_KEEMBAY,  2, 0},
                                               ConfigParams {true,  3, 5, true, 2, 5, false, CommonTestUtils::DEVICE_KEEMBAY,  2, 0},
                                              };

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         ExecNetworkGetMetricOptimalNumInferReq,
                         ::testing::ValuesIn(testConfigs),
                         ExecNetworkGetMetricOptimalNumInferReq::getTestCaseName);

TEST_P(ExecNetworkGetMetricOtherTest, modelPriority_perfHint_exclusiveAsyncReq_test) {
    unsigned int cpuOptimalNum = 3;
    unsigned int actualOptimalNum = 2;
    bool isNewAPI;
    bool actualSleep;
    std::string actualDeviceName;
    std::string performanceHint;
    IE::Parameter modelPriority;
    std::tie(isNewAPI,
             actualSleep,
             actualDeviceName,
             performanceHint,
             modelPriority) = this->GetParam();

    config.insert({InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
                   CommonTestUtils::DEVICE_CPU + std::string(",") + actualDeviceName});
    config.insert({CONFIG_KEY(PERFORMANCE_HINT), performanceHint});
    config.insert({CONFIG_KEY(MODEL_PRIORITY), modelPriority.as<std::string>()});

    if (isNewAPI) {
        ON_CALL(*core.get(), isNewAPI()).WillByDefault(Return(true));
    }
    metaDevices.push_back(
        {CommonTestUtils::DEVICE_CPU, {{CONFIG_KEY(PERFORMANCE_HINT), performanceHint}}, 3, ""});
    metaDevices.push_back({actualDeviceName, {{CONFIG_KEY(PERFORMANCE_HINT), performanceHint}}, 2, ""});

    ON_CALL(*plugin, SelectDevice(_, _, _)).WillByDefault(Return(metaDevices[1]));
    ON_CALL(*plugin, ParseMetaDevices(_, _)).WillByDefault(Return(metaDevices));
    ON_CALL(*plugin, GetValidDevice)
        .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
            std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
            return devices;
        });
    EXPECT_CALL(*plugin, ParseMetaDevices(_, _)).Times(1);
    EXPECT_CALL(*plugin, SelectDevice(_, _, _)).Times(1);

    ON_CALL(*core, GetConfig(_, StrEq(ov::compilation_num_threads.name()))).WillByDefault(Return(8));
    ON_CALL(*core,
            LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                        ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
                        ::testing::Matcher<const Config&>(_)))
        .WillByDefault(Return(cpuMockExeNetwork));

    if (actualSleep) {
        ON_CALL(*core,
                LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                            ::testing::Matcher<const std::string&>(StrEq(actualDeviceName)),
                            ::testing::Matcher<const Config&>(_)))
            .WillByDefault(InvokeWithoutArgs([this]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(10000));
                return actualMockExeNetwork;
            }));
    } else {
        ON_CALL(*core,
                LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                            ::testing::Matcher<const std::string&>(StrEq(actualDeviceName)),
                            ::testing::Matcher<const Config&>(_)))
            .WillByDefault(Return(actualMockExeNetwork));
    }

    ON_CALL(*cpuMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(RETURN_MOCK_VALUE(cpuOptimalNum));
    ON_CALL(*actualMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(RETURN_MOCK_VALUE(actualOptimalNum));

    auto AutoExecNetwork = plugin->LoadExeNetworkImpl(cnnNet, config);
    auto result = AutoExecNetwork->GetMetric(ov::hint::performance_mode.name()).as<std::string>();
    EXPECT_EQ(result, performanceHint);
    auto resPriority = AutoExecNetwork->GetMetric(ov::hint::model_priority.name());
    if (isNewAPI == true) {
        if (modelPriority.as<std::string>() == CONFIG_VALUE(MODEL_PRIORITY_LOW)) {
            EXPECT_EQ(resPriority.as<ov::hint::Priority>(), ov::hint::Priority::LOW);
        } else if (modelPriority.as<std::string>() == CONFIG_VALUE(MODEL_PRIORITY_MED)) {
            EXPECT_EQ(resPriority.as<ov::hint::Priority>(), ov::hint::Priority::MEDIUM);
        } else if (modelPriority.as<std::string>() == CONFIG_VALUE(MODEL_PRIORITY_HIGH)) {
            EXPECT_EQ(resPriority.as<ov::hint::Priority>(), ov::hint::Priority::HIGH);
        }
    } else {
        EXPECT_EQ(resPriority, modelPriority);
    }
}

const std::vector<modelPrioPerfHintTestParams> modelPrioPerfHintConfig = {
    modelPrioPerfHintTestParams{false,
                                true,
                                CommonTestUtils::DEVICE_GPU,
                                InferenceEngine::PluginConfigParams::THROUGHPUT,
                                CONFIG_VALUE(MODEL_PRIORITY_LOW)},
    modelPrioPerfHintTestParams{false,
                                true,
                                CommonTestUtils::DEVICE_GPU,
                                InferenceEngine::PluginConfigParams::LATENCY,
                                CONFIG_VALUE(MODEL_PRIORITY_LOW)},
    modelPrioPerfHintTestParams{false,
                                true,
                                CommonTestUtils::DEVICE_GPU,
                                InferenceEngine::PluginConfigParams::THROUGHPUT,
                                CONFIG_VALUE(MODEL_PRIORITY_MED)},
    modelPrioPerfHintTestParams{false,
                                true,
                                CommonTestUtils::DEVICE_GPU,
                                InferenceEngine::PluginConfigParams::LATENCY,
                                CONFIG_VALUE(MODEL_PRIORITY_MED)},
    modelPrioPerfHintTestParams{false,
                                true,
                                CommonTestUtils::DEVICE_GPU,
                                CONFIG_VALUE(THROUGHPUT),
                                CONFIG_VALUE(MODEL_PRIORITY_HIGH)},
    modelPrioPerfHintTestParams{false,
                                true,
                                CommonTestUtils::DEVICE_GPU,
                                InferenceEngine::PluginConfigParams::LATENCY,
                                CONFIG_VALUE(MODEL_PRIORITY_HIGH)},
    modelPrioPerfHintTestParams{true,
                                true,
                                CommonTestUtils::DEVICE_GPU,
                                InferenceEngine::PluginConfigParams::THROUGHPUT,
                                CONFIG_VALUE(MODEL_PRIORITY_LOW)},
    modelPrioPerfHintTestParams{true,
                                true,
                                CommonTestUtils::DEVICE_GPU,
                                InferenceEngine::PluginConfigParams::LATENCY,
                                CONFIG_VALUE(MODEL_PRIORITY_LOW)},
    modelPrioPerfHintTestParams{true,
                                true,
                                CommonTestUtils::DEVICE_GPU,
                                InferenceEngine::PluginConfigParams::THROUGHPUT,
                                CONFIG_VALUE(MODEL_PRIORITY_MED)},
    modelPrioPerfHintTestParams{true,
                                true,
                                CommonTestUtils::DEVICE_GPU,
                                InferenceEngine::PluginConfigParams::LATENCY,
                                CONFIG_VALUE(MODEL_PRIORITY_MED)},
    modelPrioPerfHintTestParams{true,
                                true,
                                CommonTestUtils::DEVICE_GPU,
                                InferenceEngine::PluginConfigParams::THROUGHPUT,
                                CONFIG_VALUE(MODEL_PRIORITY_HIGH)},
    modelPrioPerfHintTestParams{true,
                                true,
                                CommonTestUtils::DEVICE_GPU,
                                InferenceEngine::PluginConfigParams::LATENCY,
                                CONFIG_VALUE(MODEL_PRIORITY_HIGH)}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         ExecNetworkGetMetricOtherTest,
                         ::testing::ValuesIn(modelPrioPerfHintConfig),
                         ExecNetworkGetMetricOtherTest::getTestCaseName);