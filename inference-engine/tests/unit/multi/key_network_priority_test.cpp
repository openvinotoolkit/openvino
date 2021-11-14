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

using PriorityParams = std::tuple<unsigned int, std::string>; //{priority, deviceUniquName}

using ConfigParams = std::tuple<
        std::string,                        // netPrecision
        std::vector<PriorityParams>,
        >;
class KeyNetworkPriorityTest : public ::testing::TestWithParam<ConfigParams> {
public:
    std::shared_ptr<MockICore>                      core;
    std::shared_ptr<MockMultiDeviceInferencePlugin> plugin;
    std::vector<DeviceInformation>                  metaDevices;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::string netPrecision;
        std::vector<PriorityParams> PriorityConfigs;
        std::tie(netPrecision, PriorityConfigs) = obj.param;
        std::ostringstream result;
        for (auto& item : PriorityConfigs) {
            result <<  "_priority_" << std::get<0>(item);
            result <<  "_return_" << std::get<1>(item);
        }
        result << "netPrecision_" << netPrecision;
        return result.str();
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
        metaDevices.clear();
    }

    void SetUp() override {
       // prepare mockicore and cnnNetwork for loading
       core  = std::shared_ptr<MockICore>(new MockICore());
       auto* origin_plugin = new MockMultiDeviceInferencePlugin();
       plugin  = std::shared_ptr<MockMultiDeviceInferencePlugin>(origin_plugin);
       // replace core with mock Icore
       plugin->SetCore(core);
       metaDevices = {{CPU, {}, 2, "", "CPU_01"},
           {GPU, {}, 2, "", "iGPU_01"},
           {GPU, {}, 2, "", "dGPU_02"},
           {MYRIAD, {}, 2, "", "MYRIAD_01" },
           {VPUX, {}, 2, "", "VPUX_01"}};
       IE_SET_METRIC(OPTIMIZATION_CAPABILITIES, cpuAbility, {"FP32", "FP16", "INT8", "BIN"});
       IE_SET_METRIC(OPTIMIZATION_CAPABILITIES, gpuAbility, {"FP32", "FP16", "BATCHED_BLOB", "BIN"});
       IE_SET_METRIC(OPTIMIZATION_CAPABILITIES, myriadAbility, {"FP16"});
       ON_CALL(*core, GetMetric(StrEq(CPU),StrEq(METRIC_KEY(OPTIMIZATION_CAPABILITIES))))
           .WillByDefault(Return(cpuAbility));
       ON_CALL(*core, GetMetric(StrEq(GPU),StrEq(METRIC_KEY(OPTIMIZATION_CAPABILITIES))))
           .WillByDefault(Return(gpuAbility));
       ON_CALL(*core, GetMetric(StrEq(GPU),StrEq(METRIC_KEY(OPTIMIZATION_CAPABILITIES))))
           .WillByDefault(Return(myriadAbility));
    }
};

TEST_P(KeyNetworkPriorityTest, SelectDevice) {
    // get Parameter
    std::string netPrecision;
    std::vector<PriorityParams> PriorityConfigs;
    std::tie(netPrecision, PriorityConfigs) = obj.param;
    std::vector<DeviceInformation> resDevInfo;
    for (auto& item : PriorityConfigs) {
        resDevInfo.push_back(plugin->SelectDevice(metaDevices, netPrecision, std::get<0>(item)));
    }
    for(unsigned int i = 0; i < PriorityConfigs.size(); i++) {
        EXPECT_EQ(resDevInfo[i].uniqueName, std::get<1>(PriorityConfigs[i]));
        plugin->UnregisterPriority(std::get<0>(PriorityConfigs[i]), std::get<1>(PriorityConfigs[i]));
    }
}

const std::vector<ConfigParams> testConfigs = {ConfigParams {"FP32", {PriorityParams {0, "dGPU_02"},
                                                                      PriorityParams {0, "dGPU_02"},
                                                                      PriorityParams {0, "dGPU_02"}}},
                                               ConfigParams {"FP32", {PriorityParams {0, "dGPU_02"},
                                                                      PriorityParams {0, "dGPU_02"},
                                                                      PriorityParams {0, "dGPU_02"}}}
                                              };


INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, KeyNetworkPriorityTest,
                ::testing::ValuesIn(testConfigs),
            KeyNetworkPriorityTest::getTestCaseName);

