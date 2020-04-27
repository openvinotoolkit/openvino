// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"
#include <threading/ie_executor_manager.hpp>

using namespace std;
using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace {
std::string getConfigTestCaseName(testing::TestParamInfo<BehTestParams> obj) {
    std::string config_str = "";
    for (auto it = obj.param.config.cbegin(); it != obj.param.config.cend(); it++) {
        std::string v = it->second;
        std::replace(v.begin(), v.end(), '.', '_');
        config_str += it->first + "_" + v + "_";
    }
    return obj.param.device + "_" + config_str;
}
}

TEST_P(BehaviorPluginTestInferRequestConfig, CanInferWithConfig) {
    TestEnv::Ptr testEnv;
    std::map<std::string, std::string> config = GetParam().config;

    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv, config));
    sts = testEnv->inferRequest->Infer(&response);

    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
}

TEST_P(BehaviorPluginTestInferRequestConfigExclusiveAsync, canSetExclusiveAsyncRequests) {
    ASSERT_EQ(0ul, ExecutorManager::getInstance()->getExecutorsNumber());
    TestEnv::Ptr testEnv;
    std::map<std::string, std::string> config;
    config[PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS] = PluginConfigParams::YES;

    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv, config));

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

TEST_P(BehaviorPluginTestInferRequestConfigExclusiveAsync, withoutExclusiveAsyncRequests) {
    ASSERT_EQ(0u, ExecutorManager::getInstance()->getExecutorsNumber());
    TestEnv::Ptr testEnv;
    std::map<std::string, std::string> config;
    config[PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS] = PluginConfigParams::NO;

    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv, config));

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
