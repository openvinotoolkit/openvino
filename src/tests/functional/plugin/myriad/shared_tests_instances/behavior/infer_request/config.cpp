// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/config.hpp"

#include "vpu/private_plugin_config.hpp"
#include "vpu/myriad_config.hpp"

IE_SUPPRESS_DEPRECATED_START

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<std::map<std::string, std::string>> configs = {
            {}
    };

    const std::vector<std::map<std::string, std::string>> multiConfigs = {
            {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_MYRIAD}}
    };

    const std::vector<std::map<std::string, std::string>> inferConfigs = {
            {},

            {{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, CONFIG_VALUE(YES)}},
            {{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, CONFIG_VALUE(NO)}},

            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_NONE)}},
            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_ERROR)}},
            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_WARNING)}},
            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)}},
            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_DEBUG)}},
            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_TRACE)}},

            {{InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "0"}},
            {{InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "1"}},

            {{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES)}},
            {{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(NO)}},

            {{InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, CONFIG_VALUE(YES)}},
            {{InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, CONFIG_VALUE(NO)}},

            {{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "1"}},
            {{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "2"}},
            {{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "3"}},

            {{InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, CONFIG_VALUE(YES)}},
            {{InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, CONFIG_VALUE(NO)}},
    };

    const std::vector<std::map<std::string, std::string>> inferMultiConfigs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
             {CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_DEBUG)}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
             {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES)}},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestConfigTest,
                            ::testing::Combine(
                                    ::testing::Values(2u),
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                    ::testing::ValuesIn(configs)),
                            InferRequestConfigTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestConfigTest,
                            ::testing::Combine(
                                    ::testing::Values(2u),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(multiConfigs)),
                            InferRequestConfigTest::getTestCaseName);
}  // namespace
