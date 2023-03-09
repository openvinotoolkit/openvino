// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <ngraph_functions/subgraph_builders.hpp>
#include <common_test_utils/test_constants.hpp>
#include <ie_metric_helpers.hpp>
#include "mock_common.hpp"

#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "plugin/mock_auto_device_plugin.hpp"
#include "plugin/mock_infer_request.hpp"

using ::testing::Throw;
using ::testing::Matches;
using ::testing::_;
using ::testing::StrEq;
using ::testing::Return;
using ::testing::NiceMock;

using namespace MockMultiDevice;
using Config = std::map<std::string, std::string>;
using ConfigParams = std::tuple<std::vector<std::string>, std::vector<bool>, int, bool>;

class AutoRuntimeFallback : public ::testing::TestWithParam<ConfigParams> {
public:
    std::shared_ptr<ngraph::Function>               function;
    InferenceEngine::CNNNetwork                     cnnNet;
    std::shared_ptr<NiceMock<MockICore>>                      core;
    std::shared_ptr<NiceMock<MockMultiDeviceInferencePlugin>> plugin;
    // config for Auto device
    std::map<std::string, std::string>              config;
    std::vector<DeviceInformation>                 metaDevices;
    //mock exeNetwork helper
    ov::SoPtr<IExecutableNetworkInternal>  mockExeNetwork;
    ov::SoPtr<IExecutableNetworkInternal>  mockExeNetworkActual;
    ov::SoPtr<IExecutableNetworkInternal>  mockExeNetworkActualBackUp;

    std::shared_ptr<NiceMock<MockIInferRequestInternal>>     inferReqInternal;
    std::shared_ptr<NiceMock<MockIInferRequestInternal>>     inferReqInternalActual;
    std::shared_ptr<NiceMock<MockIInferRequestInternal>>     inferReqInternalActualBackUp;

    std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>>     mockIExeNet;
    std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>>     mockIExeNetActual;
    std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>>     mockIExeNetActualBackUp;

    std::shared_ptr<mockAsyncInferRequest>     mockInferrequest;
    std::shared_ptr<mockAsyncInferRequest>     mockInferrequestActual;
    std::shared_ptr<mockAsyncInferRequest>     mockInferrequestActualBackUp;

    std::shared_ptr<mockRequestExecutor>     mockExecutor;
    std::shared_ptr<mockRequestExecutor>     mockExecutorActual;
    std::shared_ptr<mockRequestExecutor>     mockExecutorActualBackUp;

    size_t optimalNum;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::vector<std::string> targetDevices;
        std::vector<bool> targetDevicesThrow;
        int loadNetworkNum;
        bool enableRumtimeFallback;
        std::tie(targetDevices, targetDevicesThrow, loadNetworkNum, enableRumtimeFallback) = obj.param;
        std::ostringstream result;
        result << "auto_runtime_fallback_";
        for (auto& device : targetDevices) {
            result << device << "_";
        }
        for (auto ifthrow : targetDevicesThrow) {
            if (ifthrow)
                result << "true_";
            else
                result << "false_";
        }
        if (enableRumtimeFallback)
            result << "enableRuntimeFallback";
        else
            result << "disableRuntimeFallback";
        return result.str();
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
        mockExeNetwork = {};
        mockExeNetworkActual = {};
        mockExeNetworkActualBackUp = {};
        config.clear();
        metaDevices.clear();
        inferReqInternal.reset();
        inferReqInternalActual.reset();
        inferReqInternalActualBackUp.reset();
        mockIExeNet.reset();
        mockIExeNetActual.reset();
        mockIExeNetActualBackUp.reset();
        mockIExeNet.reset();
        mockIExeNetActual.reset();
        mockIExeNetActualBackUp.reset();
        mockExecutor.reset();
        mockExecutorActual.reset();
        mockExecutorActualBackUp.reset();
    }

    void SetUp() override {
        // prepare mockExeNetwork
        mockIExeNet = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
        mockExeNetwork = {mockIExeNet, {}};

        mockIExeNetActual = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
        mockExeNetworkActual = {mockIExeNetActual, {}};

        mockIExeNetActualBackUp = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
        mockExeNetworkActualBackUp = {mockIExeNetActualBackUp, {}};

        // prepare mockicore and cnnNetwork for loading
        core = std::make_shared<NiceMock<MockICore>>();
        NiceMock<MockMultiDeviceInferencePlugin>* mock_multi = new NiceMock<MockMultiDeviceInferencePlugin>();
        plugin.reset(mock_multi);
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        cnnNet = InferenceEngine::CNNNetwork(function);
        plugin->SetCore(core);

        IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, supportConfigs, {});
        ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _)).WillByDefault(Return(supportConfigs));
        ON_CALL(*core, GetConfig(_, StrEq(ov::compilation_num_threads.name()))).WillByDefault(Return(12));
        std::vector<std::string> availableDevs = {"CPU", "GPU.0", "GPU.1"};
        ON_CALL(*core, GetAvailableDevices()).WillByDefault(Return(availableDevs));

        std::vector<std::string> metrics = {METRIC_KEY(SUPPORTED_CONFIG_KEYS)};
        ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_METRICS)), _)).WillByDefault(Return(metrics));

        std::vector<std::string> configKeys = {"SUPPORTED_CONFIG_KEYS", "NUM_STREAMS"};
        ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _)).WillByDefault(Return(configKeys));

        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(StrEq("GPU.0")),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetworkActual));

        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetworkActualBackUp));

        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_CPU)),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));

        ON_CALL(*plugin, ParseMetaDevices)
            .WillByDefault(
                [this](const std::string& priorityDevices, const std::map<std::string, std::string>& config) {
                    return plugin->MultiDeviceInferencePlugin::ParseMetaDevices(priorityDevices, config);
                });

        ON_CALL(*plugin, SelectDevice)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices,
                                  const std::string& netPrecision,
                                  unsigned int priority) {
                return plugin->MultiDeviceInferencePlugin::SelectDevice(metaDevices, netPrecision, priority);
            });

        ON_CALL(*plugin, GetValidDevice)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
                std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
                return devices;
            });

        ON_CALL(*plugin, GetDeviceList).WillByDefault([this](const std::map<std::string, std::string>& config) {
            return plugin->MultiDeviceInferencePlugin::GetDeviceList(config);
        });
        ON_CALL(*plugin, SelectDevice)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices,
                                  const std::string& netPrecision,
                                  unsigned int Priority) {
                return plugin->MultiDeviceInferencePlugin::SelectDevice(metaDevices, netPrecision, Priority);
            });

        inferReqInternal = std::make_shared<NiceMock<MockIInferRequestInternal>>();
        mockExecutor = std::make_shared<mockRequestExecutor>();
        IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, optimalNum, 1);
        ON_CALL(*mockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(Return(optimalNum));

        inferReqInternalActual = std::make_shared<NiceMock<MockIInferRequestInternal>>();
        mockExecutorActual = std::make_shared<mockRequestExecutor>();
        ON_CALL(*mockIExeNetActual.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(Return(optimalNum));

        inferReqInternalActualBackUp = std::make_shared<NiceMock<MockIInferRequestInternal>>();
        mockExecutorActualBackUp = std::make_shared<mockRequestExecutor>();
        ON_CALL(*mockIExeNetActualBackUp.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(Return(optimalNum));
    }
};

TEST_P(AutoRuntimeFallback, releaseResource) {
    std::string targetDev;
    std::vector<std::string> targetDevices;
    std::vector<bool> targetDevicesThrow;
    int loadNetworkNum;
    bool enableRumtimeFallback;
    std::tie(targetDevices, targetDevicesThrow, loadNetworkNum, enableRumtimeFallback) = this->GetParam();

    for (auto& deviceName : targetDevices) {
        targetDev += deviceName;
        targetDev += ((deviceName == targetDevices.back()) ? "" : ",");
    }
    plugin->SetName("AUTO");
    config.insert({InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, targetDev});
    if (!enableRumtimeFallback) {
        config.insert({{"ENABLE_RUNTIME_FALLBACK", "NO"}});
    }
    if (targetDevices.size() == 2) {
        mockInferrequest = std::make_shared<mockAsyncInferRequest>(
            inferReqInternal, mockExecutor, nullptr, targetDevicesThrow[1]);
        ON_CALL(*mockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(mockInferrequest));

        mockInferrequestActual = std::make_shared<mockAsyncInferRequest>(
            inferReqInternalActual, mockExecutorActual, nullptr, targetDevicesThrow[0]);
        ON_CALL(*mockIExeNetActual.get(), CreateInferRequest()).WillByDefault(Return(mockInferrequestActual));

        EXPECT_CALL(*core,
                    LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                                ::testing::Matcher<const std::string&>(_),
                                ::testing::Matcher<const std::map<std::string, std::string>&>(_)))
            .Times(loadNetworkNum);
    } else if (targetDevices.size() == 3) {
        mockInferrequest = std::make_shared<mockAsyncInferRequest>(
            inferReqInternal, mockExecutor, nullptr, targetDevicesThrow[2]);
        ON_CALL(*mockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(mockInferrequest));

        mockInferrequestActual = std::make_shared<mockAsyncInferRequest>(
            inferReqInternalActual, mockExecutorActual, nullptr, targetDevicesThrow[0]);
        ON_CALL(*mockIExeNetActual.get(), CreateInferRequest()).WillByDefault(Return(mockInferrequestActual));

        mockInferrequestActualBackUp = std::make_shared<mockAsyncInferRequest>(
            inferReqInternalActualBackUp, mockExecutorActualBackUp, nullptr, targetDevicesThrow[1]);
        ON_CALL(*mockIExeNetActualBackUp.get(), CreateInferRequest()).WillByDefault(Return(mockInferrequestActualBackUp));

        EXPECT_CALL(*core,
                    LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                                ::testing::Matcher<const std::string&>(_),
                                ::testing::Matcher<const std::map<std::string, std::string>&>(_)))
            .Times(loadNetworkNum);
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> exeNetwork;

    ASSERT_NO_THROW(exeNetwork = plugin->LoadExeNetworkImpl(cnnNet, config));

    std::shared_ptr<IInferRequestInternal> infer_request;
    ASSERT_NO_THROW(infer_request = exeNetwork->CreateInferRequest());
    ASSERT_NO_THROW(infer_request->StartAsync());
}

const std::vector<ConfigParams> testConfigs = {
    ConfigParams{{"GPU.0", "CPU"}, {true, true}, 3, true},
    ConfigParams{{"GPU.0", "CPU"}, {false, false}, 2, true},
    ConfigParams{{"GPU.0", "CPU"}, {true, false}, 3, true},
    ConfigParams{{"GPU.0", "CPU"}, {false, true}, 2, true},
    ConfigParams{{"GPU.0", "GPU.1", "CPU"}, {false, false, false}, 2, true},
    ConfigParams{{"GPU.0", "GPU.1", "CPU"}, {true, false, false}, 3, true},
    ConfigParams{{"GPU.0", "GPU.1", "CPU"}, {true, true, false}, 4, true},
    ConfigParams{{"GPU.0", "GPU.1", "CPU"}, {true, true, true}, 4, true},
    ConfigParams{{"GPU.0", "CPU"}, {true, true}, 2, false},
    ConfigParams{{"GPU.0", "CPU"}, {false, false}, 2, false},
    ConfigParams{{"GPU.0", "CPU"}, {true, false}, 2, false},
    ConfigParams{{"GPU.0", "CPU"}, {false, true}, 2, false},
    ConfigParams{{"GPU.0", "GPU.1", "CPU"}, {false, false, false}, 2, false},
    ConfigParams{{"GPU.0", "GPU.1", "CPU"}, {true, false, false}, 2, false},
    ConfigParams{{"GPU.0", "GPU.1", "CPU"}, {true, true, false}, 2, false},
    ConfigParams{{"GPU.0", "GPU.1", "CPU"}, {true, true, true}, 2, false},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoRuntimeFallback, AutoRuntimeFallback,
                ::testing::ValuesIn(testConfigs),
           AutoRuntimeFallback::getTestCaseName);
