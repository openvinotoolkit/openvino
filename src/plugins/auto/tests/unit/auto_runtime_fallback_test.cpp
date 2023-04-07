// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <ngraph_functions/subgraph_builders.hpp>
#include <common_test_utils/test_constants.hpp>
#include <ie_metric_helpers.hpp>
#include "include/mock_common.hpp"

#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "include/mock_auto_device_plugin.hpp"
#include "include/auto_infer_request_test_base.hpp"

using ::testing::Throw;
using ::testing::Matches;
using ::testing::_;
using ::testing::StrEq;
using ::testing::Return;
using ::testing::InvokeWithoutArgs;
using ::testing::NiceMock;

using namespace MockMultiDevice;
using Config = std::map<std::string, std::string>;
using ConfigParams = std::tuple<std::vector<std::tuple<std::string, bool>>, int, bool, bool, bool, bool>;

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
    ov::SoPtr<IExecutableNetworkInternal>  mockExeNetworkGPU_0;
    ov::SoPtr<IExecutableNetworkInternal>  mockExeNetworkGPU_1;
    ov::SoPtr<IExecutableNetworkInternal>  mockExeNetworkVPUX;

    std::shared_ptr<NiceMock<MockIInferRequestInternal>>     inferReqInternal;
    std::shared_ptr<NiceMock<MockIInferRequestInternal>>     inferReqInternalGPU_0;
    std::shared_ptr<NiceMock<MockIInferRequestInternal>>     inferReqInternalGPU_1;
    std::shared_ptr<NiceMock<MockIInferRequestInternal>>     inferReqInternalVPUX;

    std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>>     mockIExeNet;
    std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>>     mockIExeNetGPU_0;
    std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>>     mockIExeNetGPU_1;
    std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>>     mockIExeNetVPUX;

    std::shared_ptr<mockAsyncInferRequest>     mockInferrequest;
    std::shared_ptr<mockAsyncInferRequest>     mockInferrequestGPU_0;
    std::shared_ptr<mockAsyncInferRequest>     mockInferrequestGPU_1;
    std::shared_ptr<mockAsyncInferRequest>     mockInferrequestVPUX;

    std::shared_ptr<ImmediateExecutor>     mockExecutor;
    std::shared_ptr<ImmediateExecutor>     mockExecutorGPU_0;
    std::shared_ptr<ImmediateExecutor>     mockExecutorGPU_1;
    std::shared_ptr<ImmediateExecutor>     mockExecutorVPUX;

    size_t optimalNum;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::vector<std::tuple<std::string, bool>> targetDevices;
        int loadNetworkNum;
        bool enableRumtimeFallback;
        bool expectThrow;
        bool loadNetworkFail;
        bool generateWorkersFail;
        std::tie(targetDevices, loadNetworkNum, enableRumtimeFallback, expectThrow, loadNetworkFail, generateWorkersFail) = obj.param;
        std::ostringstream result;
        result << "auto_runtime_fallback_";
        for (auto deviceInfo : targetDevices) {
            std::string deviceName;
            bool ifThrow;
            std::tie(deviceName, ifThrow) = deviceInfo;
            result << deviceName << "_";
            if (ifThrow)
                result << "true_";
            else
                result << "false_";
        }
        if (enableRumtimeFallback)
            result << "enableRuntimeFallback";
        else
            result << "disableRuntimeFallback";
        if (loadNetworkFail)
            result << "loadNetworkFail";
        if (generateWorkersFail)
            result << "generateWorkersFail";
        return result.str();
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
        mockExeNetwork = {};
        mockExeNetworkGPU_0 = {};
        mockExeNetworkGPU_1 = {};
        config.clear();
        metaDevices.clear();
        inferReqInternal.reset();
        inferReqInternalGPU_0.reset();
        inferReqInternalGPU_1.reset();
        inferReqInternalVPUX.reset();
        mockIExeNet.reset();
        mockIExeNetGPU_0.reset();
        mockIExeNetGPU_1.reset();
        mockIExeNetVPUX.reset();
        mockIExeNet.reset();
        mockIExeNetGPU_0.reset();
        mockIExeNetGPU_1.reset();
        mockIExeNetVPUX.reset();
        mockExecutor.reset();
        mockExecutorGPU_0.reset();
        mockExecutorGPU_1.reset();
        mockExecutorVPUX.reset();
    }

    void SetUp() override {
        // prepare mockExeNetwork
        mockIExeNet = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
        mockExeNetwork = {mockIExeNet, {}};

        mockIExeNetGPU_0 = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
        mockExeNetworkGPU_0 = {mockIExeNetGPU_0, {}};

        mockIExeNetGPU_1 = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
        mockExeNetworkGPU_1 = {mockIExeNetGPU_1, {}};

        mockIExeNetVPUX = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
        mockExeNetworkVPUX = {mockIExeNetVPUX, {}};

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
        std::vector<std::string> availableDevs = {"CPU", "GPU.0", "GPU.1", "VPUX"};
        ON_CALL(*core, GetAvailableDevices()).WillByDefault(Return(availableDevs));

        std::vector<std::string> metrics = {METRIC_KEY(SUPPORTED_CONFIG_KEYS)};
        ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_METRICS)), _)).WillByDefault(Return(metrics));

        std::vector<std::string> configKeys = {"SUPPORTED_CONFIG_KEYS", "NUM_STREAMS"};
        ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _)).WillByDefault(Return(configKeys));

        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(StrEq("GPU.0")),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(InvokeWithoutArgs([this]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        return mockExeNetworkGPU_0; }));
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(InvokeWithoutArgs([this]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        return mockExeNetworkGPU_1; }));
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(StrEq(CommonTestUtils::DEVICE_KEEMBAY)),
                    ::testing::Matcher<const Config&>(_))).WillByDefault(InvokeWithoutArgs([this]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        return mockExeNetworkVPUX; }));

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
            .WillByDefault([](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
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
        mockExecutor = std::make_shared<ImmediateExecutor>();
        IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, optimalNum, 1);
        ON_CALL(*mockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(Return(optimalNum));

        inferReqInternalGPU_0 = std::make_shared<NiceMock<MockIInferRequestInternal>>();
        mockExecutorGPU_0 = std::make_shared<ImmediateExecutor>();
        ON_CALL(*mockIExeNetGPU_0.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(Return(optimalNum));

        inferReqInternalGPU_1 = std::make_shared<NiceMock<MockIInferRequestInternal>>();
        mockExecutorGPU_1 = std::make_shared<ImmediateExecutor>();
        ON_CALL(*mockIExeNetGPU_1.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(Return(optimalNum));

        inferReqInternalVPUX = std::make_shared<NiceMock<MockIInferRequestInternal>>();
        mockExecutorVPUX = std::make_shared<ImmediateExecutor>();
        ON_CALL(*mockIExeNetVPUX.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(Return(optimalNum));
    }
};

using AutoCTPUTRuntimeFallback = AutoRuntimeFallback;

TEST_P(AutoRuntimeFallback, releaseResource) {
    std::string targetDev;
    std::vector<std::tuple<std::string, bool>> targetDevices;
    int loadNetworkNum;
    bool enableRumtimeFallback;
    bool expectThrow;
    bool loadNetworkFail;
    bool generateWorkersFail;
    std::tie(targetDevices, loadNetworkNum, enableRumtimeFallback, expectThrow, loadNetworkFail, generateWorkersFail) = this->GetParam();
    if (loadNetworkFail) {
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
            ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
            ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));
    }
    for (auto& deviceInfo : targetDevices) {
        std::string deviceName;
        bool ifThrow;
        std::tie(deviceName, ifThrow) = deviceInfo;
        targetDev += deviceName;
        targetDev += ((deviceInfo == targetDevices.back()) ? "" : ",");
        if (deviceName == "CPU") {
            mockInferrequest = std::make_shared<mockAsyncInferRequest>(
                inferReqInternal, mockExecutor, nullptr, ifThrow);
            ON_CALL(*mockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(mockInferrequest));
        } else if (deviceName == "GPU.0") {
            mockInferrequestGPU_0 = std::make_shared<mockAsyncInferRequest>(
                inferReqInternalGPU_0, mockExecutorGPU_0, nullptr, ifThrow);
            ON_CALL(*mockIExeNetGPU_0.get(), CreateInferRequest()).WillByDefault(InvokeWithoutArgs([this]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(0));
                        return mockInferrequestGPU_0; }));
        } else if (deviceName == "GPU.1") {
            if (generateWorkersFail) {
                mockInferrequestGPU_1 = std::make_shared<mockAsyncInferRequest>(
                    inferReqInternalGPU_1, mockExecutorGPU_1, nullptr, ifThrow);
                ON_CALL(*mockIExeNetGPU_1.get(), CreateInferRequest()).WillByDefault(Throw(InferenceEngine::GeneralError{""}));
            } else {
                mockInferrequestGPU_1 = std::make_shared<mockAsyncInferRequest>(
                    inferReqInternalGPU_1, mockExecutorGPU_1, nullptr, ifThrow);
                ON_CALL(*mockIExeNetGPU_1.get(), CreateInferRequest()).WillByDefault(InvokeWithoutArgs([this]() {
                            std::this_thread::sleep_for(std::chrono::milliseconds(0));
                            return mockInferrequestGPU_1; }));
            }
        } else if (deviceName == "VPUX") {
            mockInferrequestVPUX = std::make_shared<mockAsyncInferRequest>(
                inferReqInternalVPUX, mockExecutorVPUX, nullptr, ifThrow);
            ON_CALL(*mockIExeNetVPUX.get(), CreateInferRequest()).WillByDefault(InvokeWithoutArgs([this]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(0));
                        return mockInferrequestVPUX; }));
        } else {
            return;
        }
    }
    plugin->SetName("AUTO");
    config.insert({InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, targetDev});
    if (!enableRumtimeFallback) {
        config.insert({{"ENABLE_RUNTIME_FALLBACK", "NO"}});
    }

    EXPECT_CALL(*core,
                LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                            ::testing::Matcher<const std::string&>(_),
                            ::testing::Matcher<const std::map<std::string, std::string>&>(_)))
        .Times(loadNetworkNum);

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> exeNetwork;
    std::shared_ptr<IInferRequestInternal> infer_request;

    ASSERT_NO_THROW(exeNetwork = plugin->LoadExeNetworkImpl(cnnNet, config));
    ASSERT_NO_THROW(infer_request = exeNetwork->CreateInferRequest());
    if (expectThrow) {
        EXPECT_THROW(infer_request->Infer(), IE::Exception);
    } else {
        ASSERT_NO_THROW(infer_request->Infer());
    }
}

const std::vector<ConfigParams> testConfigs = {
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}}, 2, true, true, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}}, 2, true, false, false, false},
    ConfigParams{{{"GPU.0", false}, {"GPU.1", true}}, 1, true, false, false, false},
    ConfigParams{{{"GPU.0", false}, {"GPU.1", false}}, 1, true, false, false, false},
    //CPU_HELP does not throw
    ConfigParams{{{"GPU.0", false}, {"CPU", false}}, 2, true, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"CPU", false}}, 2, true, false, false, false},
    //CPU_HELP throw
    ConfigParams{{{"GPU.0", false}, {"CPU", true}}, 2, true, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"CPU", true}}, 2, true, true, false, false},
    // 3 devices
    ConfigParams{{{"GPU.0", false}, {"GPU.1", false}, {"VPUX", false}}, 1, true, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}, {"VPUX", false}}, 2, true, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}, {"VPUX", false}}, 3, true, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}, {"VPUX", true}}, 3, true, true, false, false},
    //CPU_HELP does not throw
    ConfigParams{{{"GPU.0", false}, {"GPU.1", false}, {"CPU", false}}, 2, true, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}, {"CPU", false}}, 2, true, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}, {"CPU", false}}, 2, true, false, false, false},
    //CPU_HELP throw
    ConfigParams{{{"GPU.0", false}, {"GPU.1", false}, {"CPU", true}}, 2, true, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}, {"CPU", true}}, 3, true, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}, {"CPU", true}}, 3, true, true, false, false},
    // disable RumtimeFallback
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}}, 1, false, true, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}}, 1, false, true, false, false},
    ConfigParams{{{"GPU.0", false}, {"GPU.1", true}}, 1, false, false, false, false},
    ConfigParams{{{"GPU.0", false}, {"GPU.1", false}}, 1, false, false, false, false},
    //CPU_HELP does not throw
    ConfigParams{{{"GPU.0", false}, {"CPU", false}}, 2, false, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"CPU", false}}, 2, false, false, false, false},
    //CPU_HELP throw
    ConfigParams{{{"GPU.0", false}, {"CPU", true}}, 2, false, true, false, false},
    ConfigParams{{{"GPU.0", true}, {"CPU", true}}, 2, false, true, false, false},
    // 3 devices
    ConfigParams{{{"GPU.0", false}, {"GPU.1", false}, {"VPUX", false}}, 1, false, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}, {"VPUX", false}}, 1, false, true, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}, {"VPUX", false}}, 1, false, true, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}, {"VPUX", true}}, 1, false, true, false, false},
    //CPU_HELP does not throw
    ConfigParams{{{"GPU.0", false}, {"GPU.1", false}, {"CPU", false}}, 2, false, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}, {"CPU", false}}, 2, false, false, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}, {"CPU", false}}, 2, false, false, false, false},
    //CPU_HELP throw
    ConfigParams{{{"GPU.0", false}, {"GPU.1", false}, {"CPU", true}}, 2, false, true, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}, {"CPU", true}}, 2, false, true, false, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", true}, {"CPU", true}}, 2, false, true, false, false},
    // loadFail and CreateInferRequestFail
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}, {"VPUX", false}}, 3, true, false, true, false},
    ConfigParams{{{"GPU.0", true}, {"GPU.1", false}, {"VPUX", false}}, 3, true, false, false, true},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoRuntimeFallback, AutoRuntimeFallback,
                ::testing::ValuesIn(testConfigs),
           AutoRuntimeFallback::getTestCaseName);

TEST_P(AutoCTPUTRuntimeFallback, ctputDeviceInferFailTest) {
    std::string targetDev;
    std::vector<std::tuple<std::string, bool>> targetDevices; //std::tuple<deviceName, will infer throw exception>
    int loadNetworkNum;
    bool enableRumtimeFallback;
    bool expectThrow;
    bool loadNetworkFail;
    bool generateWorkersFail;
    std::tie(targetDevices, loadNetworkNum, enableRumtimeFallback, expectThrow, loadNetworkFail, generateWorkersFail) = this->GetParam();
    if (loadNetworkFail) {
        ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
            ::testing::Matcher<const std::string&>(StrEq("GPU.1")),
            ::testing::Matcher<const Config&>(_))).WillByDefault(Throw(InferenceEngine::GeneralError{""}));
    }
    for (auto& deviceInfo : targetDevices) {
        std::string deviceName;
        bool ifThrow;
        std::tie(deviceName, ifThrow) = deviceInfo;
        targetDev += deviceName;
        targetDev += ((deviceInfo == targetDevices.back()) ? "" : ",");
        if (deviceName == "CPU") {
            mockInferrequest = std::make_shared<mockAsyncInferRequest>(
                inferReqInternal, mockExecutor, nullptr, ifThrow);
            ON_CALL(*mockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(mockInferrequest));
        } else if (deviceName == "GPU.0") {
            mockInferrequestGPU_0 = std::make_shared<mockAsyncInferRequest>(
                inferReqInternalGPU_0, mockExecutorGPU_0, nullptr, ifThrow);
            ON_CALL(*mockIExeNetGPU_0.get(), CreateInferRequest()).WillByDefault(InvokeWithoutArgs([this]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(0));
                        return mockInferrequestGPU_0; }));
        } else if (deviceName == "GPU.1") {
            if (generateWorkersFail) {
                mockInferrequestGPU_1 =
                    std::make_shared<mockAsyncInferRequest>(inferReqInternalGPU_1, mockExecutorGPU_1, nullptr, ifThrow);
                ON_CALL(*mockIExeNetGPU_1.get(), CreateInferRequest())
                    .WillByDefault(Throw(InferenceEngine::GeneralError{""}));
            } else {
                mockInferrequestGPU_1 =
                    std::make_shared<mockAsyncInferRequest>(inferReqInternalGPU_1, mockExecutorGPU_1, nullptr, ifThrow);
                ON_CALL(*mockIExeNetGPU_1.get(), CreateInferRequest()).WillByDefault(InvokeWithoutArgs([this]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(0));
                    return mockInferrequestGPU_1;
                }));
            }
        } else {
            return;
        }
    }
    plugin->SetName("AUTO");
    config.insert({InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, targetDev});
    config.insert({InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT,
                   InferenceEngine::PluginConfigParams::CUMULATIVE_THROUGHPUT});
    if (!enableRumtimeFallback) {
        config.insert({{"ENABLE_RUNTIME_FALLBACK", "NO"}});
    }

    EXPECT_CALL(*core,
                LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                            ::testing::Matcher<const std::string&>(_),
                            ::testing::Matcher<const std::map<std::string, std::string>&>(_)))
        .Times(loadNetworkNum);

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> exeNetwork;
    std::shared_ptr<IInferRequestInternal> infer_request;

    ASSERT_NO_THROW(exeNetwork = plugin->LoadExeNetworkImpl(cnnNet, config));
    ASSERT_NO_THROW(infer_request = exeNetwork->CreateInferRequest());
    if (expectThrow) {
        EXPECT_THROW(infer_request->Infer(), IE::Exception);
    } else {
        ASSERT_NO_THROW(infer_request->Infer());
    }
}

// ConfigParams: targetDevices(deviceName, will infer throw exception), loadNetworkNum, enableRumtimeFallback,
// expectThrow, loadNetworkFail, generateWorkersFail
const std::vector<ConfigParams> testCtputConfigs = {
    ConfigParams{{{"CPU", false}, {"GPU.0", true}, {"GPU.1", true}}, 3, true, false, false, false},
    ConfigParams{{{"CPU", true}, {"GPU.0", false}, {"GPU.1", true}}, 3, true, false, false, false},
    ConfigParams{{{"CPU", true}, {"GPU.0", true}, {"GPU.1", true}}, 3, true, true, false, false},
    // disable RumtimeFallback
    ConfigParams{{{"CPU", false}, {"GPU.0", false}, {"GPU.1", false}}, 3, false, false, false, false},
    ConfigParams{{{"CPU", true}, {"GPU.0", false}, {"GPU.1", false}}, 3, false, true, false, false},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoCTPUTRuntimeFallback,
                         AutoCTPUTRuntimeFallback,
                         ::testing::ValuesIn(testCtputConfigs),
                         AutoCTPUTRuntimeFallback::getTestCaseName);
