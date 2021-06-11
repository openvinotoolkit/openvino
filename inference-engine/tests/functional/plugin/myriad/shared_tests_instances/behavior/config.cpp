// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/vpu_plugin_config.hpp"
#include "vpu/private_plugin_config.hpp"
#include "behavior/config.hpp"

IE_SUPPRESS_DEPRECATED_START

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> Configs = {
            {{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, CONFIG_VALUE(YES)}},
            {{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, CONFIG_VALUE(NO)}},

            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_NONE)}},
            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_ERROR)}},
            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_WARNING)}},
            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)}},
            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_DEBUG)}},
            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_TRACE)}},

            {{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES)}},
            {{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(NO)}},

            {{InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "-1"}},
            {{InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "0"}},
            {{InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "10"}},

            {{InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, CONFIG_VALUE(YES)}},
            {{InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, CONFIG_VALUE(NO)}},
            {{InferenceEngine::MYRIAD_PROTOCOL, InferenceEngine::MYRIAD_USB}},
            {{InferenceEngine::MYRIAD_PROTOCOL, InferenceEngine::MYRIAD_PCIE}},

            {{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "1"}},
            {{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "2"}},
            {{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "3"}},

            {{InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, CONFIG_VALUE(YES)}},
            {{InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, CONFIG_VALUE(NO)}},

            // Deprecated
            {{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), CONFIG_VALUE(YES)}},
            {{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), CONFIG_VALUE(NO)}},

            {{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(YES)}},
            {{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(NO)}},

            {{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), CONFIG_VALUE(YES)}},
            {{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), CONFIG_VALUE(NO)}},

            {{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), VPU_MYRIAD_CONFIG_VALUE(USB)}},
            {{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), VPU_MYRIAD_CONFIG_VALUE(PCIE)}},

            {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2450)}},
            {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2480)}}
    };

    const std::vector<std::map<std::string, std::string>> MultiConfigs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
             {CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_DEBUG)}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
             {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES)}},

            // Deprecated
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
             {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(YES)}}
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, CorrectConfigTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                    ::testing::ValuesIn(Configs)),
                            CorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, CorrectConfigTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(MultiConfigs)),
                            CorrectConfigTests::getTestCaseName);

    const std::vector<std::map<std::string, std::string>> inconfigs = {
            {{InferenceEngine::MYRIAD_PROTOCOL, "BLUETOOTH"}},
            {{InferenceEngine::MYRIAD_PROTOCOL, "LAN"}},

            {{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, "ON"}},
            {{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, "OFF"}},

            {{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, "ON"}},
            {{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, "OFF"}},

            {{CONFIG_KEY(LOG_LEVEL), "VERBOSE"}},

            {{InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "-10"}},

            {{InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, "ON"}},
            {{InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, "OFF"}},

            {{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "Two"}},
            {{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "SINGLE"}},

            {{InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, "ON"}},
            {{InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, "OFF"}},

            // Deprecated
            {{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), "BLUETOOTH"}},
            {{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), "LAN"}},

            {{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), "ON"}},
            {{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), "OFF"}},

            {{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), "ON"}},
            {{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), "OFF"}},

            {{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), "ON"}},
            {{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), "OFF"}},

            {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "-1"}},
            {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "0"}},
            {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "1"}},
    };

    const std::vector<std::map<std::string, std::string>> multiinconfigs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
             {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, "ON"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
             {CONFIG_KEY(LOG_LEVEL), "VERBOSE"}},

            // Deprecated
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
             {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), "ON"}},

            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
             {VPU_MYRIAD_CONFIG_KEY(PLATFORM), "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
             {VPU_MYRIAD_CONFIG_KEY(PLATFORM), "0"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
             {VPU_MYRIAD_CONFIG_KEY(PLATFORM), "1"}},
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, IncorrectConfigTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                    ::testing::ValuesIn(inconfigs)),
                            IncorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, IncorrectConfigTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(multiinconfigs)),
                            IncorrectConfigTests::getTestCaseName);



    const std::vector<std::map<std::string, std::string>> Inconf = {
            {{"some_nonexistent_key", "some_unknown_value"}}
    };

    const std::vector<std::map<std::string, std::string>> multiInconf = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_MYRIAD},
             {"some_nonexistent_key", "some_unknown_value"}}
    };


    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, IncorrectConfigAPITests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                    ::testing::ValuesIn(Inconf)),
                            IncorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, IncorrectConfigAPITests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(multiInconf)),
                            IncorrectConfigAPITests::getTestCaseName);




    const std::vector<std::map<std::string, std::string>> conf = {
            {}
    };

    const std::vector<std::map<std::string, std::string>> multiconf = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_MYRIAD}}
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, CorrectConfigAPITests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                    ::testing::ValuesIn(conf)),
                            CorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, CorrectConfigAPITests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(multiconf)),
                            CorrectConfigAPITests::getTestCaseName);
} // namespace
