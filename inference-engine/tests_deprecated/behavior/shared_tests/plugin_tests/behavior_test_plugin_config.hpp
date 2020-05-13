// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"
#include <threading/ie_executor_manager.hpp>
#include <ie_core.hpp>

using namespace std;
using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace {
    std::ostream &operator<<(std::ostream &os, const BehTestParams &p) {
        return os << "#";
    }

    std::string getTestCaseName(testing::TestParamInfo<BehTestParams> obj) {
        std::string config_str = "";
        for (auto it = obj.param.config.cbegin(); it != obj.param.config.cend(); it++) {
            std::string v = it->second;
            std::replace(v.begin(), v.end(), '.', '_');
            config_str += it->first + "_" + v + "_";
        }
        return obj.param.device + "_" + config_str;
    }
}

// Setting empty config doesn't throw
TEST_P(BehaviorPluginCorrectConfigTest, SetEmptyConfig) {
    InferenceEngine::Core core;
    std::map<std::string, std::string> config;
    const std::string device = GetParam().device;
    ASSERT_NO_THROW(core.GetMetric(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_NO_THROW(core.SetConfig(config, GetParam().device));
}

// Setting correct config doesn't throw
TEST_P(BehaviorPluginCorrectConfigTest, SetCorrectConfig) {
    InferenceEngine::Core core;
    std::map<std::string, std::string> config = GetParam().config;
    const std::string device = GetParam().device;
    ASSERT_NO_THROW(core.GetMetric(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    ASSERT_NO_THROW(core.SetConfig(config, GetParam().device));
}

TEST_P(BehaviorPluginIncorrectConfigTest, SetConfigWithIncorrectKey) {
    InferenceEngine::Core core;
    std::map<std::string, std::string> config = GetParam().config;
    const std::string device = GetParam().device;
    if (device.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
        device.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
        ASSERT_NO_THROW(core.GetMetric(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
        ASSERT_THROW(core.SetConfig(config, GetParam().device), InferenceEngineException);
    } else {
        ASSERT_NO_THROW(core.GetMetric(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
        ASSERT_NO_THROW(core.SetConfig(config, GetParam().device));
    }
}

TEST_P(BehaviorPluginIncorrectConfigTest, canNotLoadNetworkWithIncorrectConfig) {
    auto param = GetParam();
    std::map<std::string, std::string> config = param.config;
    InferenceEngine::Core core;
    IExecutableNetwork::Ptr exeNetwork;
    CNNNetwork cnnNetwork = core.ReadNetwork(GetParam().model_xml_str, GetParam().weights_blob);

    ASSERT_THROW(exeNetwork = core.LoadNetwork(cnnNetwork, param.device, config), InferenceEngineException);
}

TEST_P(BehaviorPluginIncorrectConfigTestInferRequestAPI, SetConfigWithNoExistingKey) {
    std::string refError = NOT_FOUND_str;
    InferenceEngine::Core core;
    std::map<std::string, std::string> config = GetParam().config;
    const std::string device = GetParam().device;
    ASSERT_NO_THROW(core.GetMetric(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    if (device.find(CommonTestUtils::DEVICE_GNA) != std::string::npos) {
        ASSERT_THROW(core.SetConfig(config, GetParam().device), NotFound);
    } else {
        try {
            core.SetConfig(config, GetParam().device);
        } catch (InferenceEngineException ex) {
            ASSERT_STR_CONTAINS(ex.what(), refError);
        }
    }
}

IE_SUPPRESS_DEPRECATED_START

TEST_P(BehaviorPluginCorrectConfigTestInferRequestAPI, canSetExclusiveAsyncRequests) {
    ASSERT_EQ(0u, ExecutorManager::getInstance()->getExecutorsNumber());
    auto param = GetParam();
    InferenceEngine::Core core;
    std::map<std::string, std::string> config = {{KEY_EXCLUSIVE_ASYNC_REQUESTS, YES}};
    config.insert(param.config.begin(), param.config.end());

    const std::string device = GetParam().device;
    if (device.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
        device.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
        ASSERT_NO_THROW(core.SetConfig(config, GetParam().device));
    }

    CNNNetwork cnnNetwork = core.ReadNetwork(GetParam().model_xml_str, GetParam().weights_blob);

    ExecutableNetwork exeNetwork = core.LoadNetwork(cnnNetwork, GetParam().device, config);
    exeNetwork.CreateInferRequest();

    // TODO: there is no executors to sync. should it be supported natively in HDDL API?
    if (GetParam().device == CommonTestUtils::DEVICE_HDDL) {
        ASSERT_EQ(0u, ExecutorManager::getInstance()->getExecutorsNumber());
    } else if (GetParam().device == CommonTestUtils::DEVICE_FPGA) {
        ASSERT_EQ(2u, ExecutorManager::getInstance()->getExecutorsNumber());
    } else if (GetParam().device == CommonTestUtils::DEVICE_MYRIAD) {
        ASSERT_EQ(2u, ExecutorManager::getInstance()->getExecutorsNumber());
    } else if (GetParam().device == CommonTestUtils::DEVICE_KEEMBAY) {
        ASSERT_EQ(2u, ExecutorManager::getInstance()->getExecutorsNumber());
    } else if (GetParam().device == CommonTestUtils::DEVICE_GNA) {
        ASSERT_EQ(0u, ExecutorManager::getInstance()->getExecutorsNumber());
    } else if (GetParam().device == CommonTestUtils::DEVICE_MULTI) {
        // for multi-device the number of Executors is not known (defined by the devices configuration)
    } else {
        ASSERT_EQ(1u, ExecutorManager::getInstance()->getExecutorsNumber());
    }
}

TEST_P(BehaviorPluginCorrectConfigTestInferRequestAPI, withoutExclusiveAsyncRequests) {
    ASSERT_EQ(0u, ExecutorManager::getInstance()->getExecutorsNumber());

    auto param = GetParam();
    InferenceEngine::Core core;

    std::map<std::string, std::string> config = {{KEY_EXCLUSIVE_ASYNC_REQUESTS, NO}};
    config.insert(param.config.begin(), param.config.end());

    const std::string device = GetParam().device;
    if (device.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
        device.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
        ASSERT_NO_THROW(core.SetConfig(config, param.device));
    }

    CNNNetwork cnnNetwork = core.ReadNetwork(param.model_xml_str, param.weights_blob);

    ExecutableNetwork exeNetwork = core.LoadNetwork(cnnNetwork, param.device, config);
    exeNetwork.CreateInferRequest();


    if (GetParam().device == CommonTestUtils::DEVICE_FPGA) {
        ASSERT_EQ(1u, ExecutorManager::getInstance()->getExecutorsNumber());
    } else if (GetParam().device == CommonTestUtils::DEVICE_MYRIAD) {
        ASSERT_EQ(1u, ExecutorManager::getInstance()->getExecutorsNumber());
    } else if (GetParam().device == CommonTestUtils::DEVICE_KEEMBAY) {
        ASSERT_EQ(1u, ExecutorManager::getInstance()->getExecutorsNumber());
    } else if (GetParam().device == CommonTestUtils::DEVICE_MULTI) {
        // for multi-device the number of Executors is not known (defined by the devices configuration)
    } else {
        ASSERT_EQ(0u, ExecutorManager::getInstance()->getExecutorsNumber());
    }
}

TEST_P(BehaviorPluginCorrectConfigTestInferRequestAPI, reusableCPUStreamsExecutor) {
    ASSERT_EQ(0u, ExecutorManager::getInstance()->getExecutorsNumber());
    ASSERT_EQ(0u, ExecutorManager::getInstance()->getIdleCPUStreamsExecutorsNumber());

    auto param = GetParam();
    InferenceEngine::Core core;
    {
        std::map<std::string, std::string> config = {{KEY_EXCLUSIVE_ASYNC_REQUESTS, NO}};
        config.insert(param.config.begin(), param.config.end());

        const std::string device = GetParam().device;
        if (device.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
            device.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
            ASSERT_NO_THROW(core.SetConfig(config, param.device));
        }

        CNNNetwork cnnNetwork = core.ReadNetwork(param.model_xml_str, param.weights_blob);

        ExecutableNetwork exeNetwork = core.LoadNetwork(cnnNetwork, param.device, config);
        exeNetwork.CreateInferRequest();


        if (GetParam().device == CommonTestUtils::DEVICE_FPGA) {
            ASSERT_EQ(1u, ExecutorManager::getInstance()->getExecutorsNumber());
            ASSERT_EQ(0u, ExecutorManager::getInstance()->getIdleCPUStreamsExecutorsNumber());
        } else if (GetParam().device == CommonTestUtils::DEVICE_MYRIAD) {
            ASSERT_EQ(1u, ExecutorManager::getInstance()->getExecutorsNumber());
            ASSERT_EQ(0u, ExecutorManager::getInstance()->getIdleCPUStreamsExecutorsNumber());
        } else if (GetParam().device == CommonTestUtils::DEVICE_KEEMBAY) {
            ASSERT_EQ(1u, ExecutorManager::getInstance()->getExecutorsNumber());
            ASSERT_EQ(0u, ExecutorManager::getInstance()->getIdleCPUStreamsExecutorsNumber());
        } else if (GetParam().device == CommonTestUtils::DEVICE_MULTI) {
            // for multi-device the number of Executors is not known (defined by the devices configuration)
        } else {
            ASSERT_EQ(0u, ExecutorManager::getInstance()->getExecutorsNumber());
            ASSERT_GE(2u, ExecutorManager::getInstance()->getIdleCPUStreamsExecutorsNumber());
        }
  }
    if (GetParam().device == CommonTestUtils::DEVICE_CPU) {
        ASSERT_NE(0u, ExecutorManager::getInstance()->getIdleCPUStreamsExecutorsNumber());
        ASSERT_NO_THROW(core.UnregisterPlugin("CPU"));
        ASSERT_EQ(0u, ExecutorManager::getInstance()->getExecutorsNumber());
        ASSERT_EQ(0u, ExecutorManager::getInstance()->getIdleCPUStreamsExecutorsNumber());
    }
}
