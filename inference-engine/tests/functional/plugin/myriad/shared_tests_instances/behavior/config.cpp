// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/vpu_plugin_config.hpp"
#include "vpu/private_plugin_config.hpp"
#include "behavior/config.hpp"
#include "myriad_devices.hpp"

IE_SUPPRESS_DEPRECATED_START

namespace {

using namespace BehaviorTestsDefinitions;
using namespace InferenceEngine::PluginConfigParams;

const std::vector<InferenceEngine::Precision>& getPrecisions() {
    static const std::vector<InferenceEngine::Precision> precisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
    };
    return precisions;
}

std::vector<std::map<std::string, std::string>> getCorrectConfigs() {
    std::vector<std::map<std::string, std::string>> correctConfigs = {
        {{KEY_LOG_LEVEL, LOG_NONE}},
        {{KEY_LOG_LEVEL, LOG_ERROR}},
        {{KEY_LOG_LEVEL, LOG_WARNING}},
        {{KEY_LOG_LEVEL, LOG_INFO}},
        {{KEY_LOG_LEVEL, LOG_DEBUG}},
        {{KEY_LOG_LEVEL, LOG_TRACE}},

        {{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "-1"}},
        {{InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "0"}},
        {{InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "10"}},

        {{InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, CONFIG_VALUE(NO)}},

        {{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "1"}},
        {{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "2"}},
        {{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "3"}},

        {{InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, CONFIG_VALUE(YES)}},
        {{InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, CONFIG_VALUE(NO)}},

        // Deprecated
        {{VPU_CONFIG_KEY(LOG_LEVEL), LOG_NONE}},
        {{VPU_CONFIG_KEY(LOG_LEVEL), LOG_ERROR}},
        {{VPU_CONFIG_KEY(LOG_LEVEL), LOG_WARNING}},
        {{VPU_CONFIG_KEY(LOG_LEVEL), LOG_INFO}},
        {{VPU_CONFIG_KEY(LOG_LEVEL), LOG_DEBUG}},
        {{VPU_CONFIG_KEY(LOG_LEVEL), LOG_TRACE}},

        {{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), CONFIG_VALUE(YES)}},
        {{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), CONFIG_VALUE(NO)}},

        {{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(YES)}},
        {{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(NO)}},

        {{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), CONFIG_VALUE(YES)}},
        {{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), CONFIG_VALUE(NO)}},

        {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2480)}},

        {
            {KEY_LOG_LEVEL, LOG_INFO},
            {InferenceEngine::MYRIAD_COPY_OPTIMIZATION, InferenceEngine::PluginConfigParams::NO},
            {InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "10"},
            {InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "1"},
            {InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, CONFIG_VALUE(YES)},
        },
    };

    MyriadDevicesInfo info;
    if (info.getAmountOfDevices(ncDeviceProtocol_t::NC_PCIE) > 0) {
        correctConfigs.emplace_back(std::map<std::string, std::string>{{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), VPU_MYRIAD_CONFIG_VALUE(PCIE)}});
        correctConfigs.emplace_back(std::map<std::string, std::string>{{InferenceEngine::MYRIAD_PROTOCOL, InferenceEngine::MYRIAD_PCIE}});
    }

    if (info.getAmountOfDevices(ncDeviceProtocol_t::NC_USB) > 0) {
        correctConfigs.emplace_back(std::map<std::string, std::string>{{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), VPU_MYRIAD_CONFIG_VALUE(USB)}});
        correctConfigs.emplace_back(std::map<std::string, std::string>{{InferenceEngine::MYRIAD_PROTOCOL, InferenceEngine::MYRIAD_USB}});
    }

    return correctConfigs;
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, CorrectConfigTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::ValuesIn(getCorrectConfigs())),
    CorrectConfigTests::getTestCaseName);

const std::vector<std::map<std::string, std::string>>& getCorrectMultiConfigs() {
    static const std::vector<std::map<std::string, std::string>> correctMultiConfigs = {
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {KEY_LOG_LEVEL, LOG_DEBUG},
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {InferenceEngine::MYRIAD_COPY_OPTIMIZATION, InferenceEngine::PluginConfigParams::NO},
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, YES},
        },

        // Deprecated
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_CONFIG_KEY(LOG_LEVEL), LOG_DEBUG},
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(YES)},
        },
    };
    return correctMultiConfigs;
}

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, CorrectConfigTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MULTI),
        ::testing::ValuesIn(getCorrectMultiConfigs())),
    CorrectConfigTests::getTestCaseName);

const std::vector<std::pair<std::string, InferenceEngine::Parameter>>& getDefaultEntries() {
    static const std::vector<std::pair<std::string, InferenceEngine::Parameter>> defaultEntries = {
        {KEY_LOG_LEVEL, {LOG_NONE}},
    };
    return defaultEntries;
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, CorrectSingleOptionDefaultValueConfigTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::ValuesIn(getDefaultEntries())));

const std::vector<std::tuple<std::string, std::string, InferenceEngine::Parameter>>& getCustomEntries() {
    static const std::vector<std::tuple<std::string, std::string, InferenceEngine::Parameter>> customEntries = {
        std::make_tuple(KEY_LOG_LEVEL, LOG_NONE,    InferenceEngine::Parameter{LOG_NONE}),
        std::make_tuple(KEY_LOG_LEVEL, LOG_ERROR,   InferenceEngine::Parameter{LOG_ERROR}),
        std::make_tuple(KEY_LOG_LEVEL, LOG_WARNING, InferenceEngine::Parameter{LOG_WARNING}),
        std::make_tuple(KEY_LOG_LEVEL, LOG_INFO,    InferenceEngine::Parameter{LOG_INFO}),
        std::make_tuple(KEY_LOG_LEVEL, LOG_DEBUG,   InferenceEngine::Parameter{LOG_DEBUG}),
        std::make_tuple(KEY_LOG_LEVEL, LOG_TRACE,   InferenceEngine::Parameter{LOG_TRACE}),

        std::make_tuple(VPU_CONFIG_KEY(LOG_LEVEL), LOG_NONE,    InferenceEngine::Parameter{LOG_NONE}),
        std::make_tuple(VPU_CONFIG_KEY(LOG_LEVEL), LOG_ERROR,   InferenceEngine::Parameter{LOG_ERROR}),
        std::make_tuple(VPU_CONFIG_KEY(LOG_LEVEL), LOG_WARNING, InferenceEngine::Parameter{LOG_WARNING}),
        std::make_tuple(VPU_CONFIG_KEY(LOG_LEVEL), LOG_INFO,    InferenceEngine::Parameter{LOG_INFO}),
        std::make_tuple(VPU_CONFIG_KEY(LOG_LEVEL), LOG_DEBUG,   InferenceEngine::Parameter{LOG_DEBUG}),
        std::make_tuple(VPU_CONFIG_KEY(LOG_LEVEL), LOG_TRACE,   InferenceEngine::Parameter{LOG_TRACE}),

        std::make_tuple(InferenceEngine::MYRIAD_COPY_OPTIMIZATION, InferenceEngine::PluginConfigParams::YES, InferenceEngine::Parameter{true}),
        std::make_tuple(InferenceEngine::MYRIAD_COPY_OPTIMIZATION, InferenceEngine::PluginConfigParams::NO,  InferenceEngine::Parameter{false}),
    };
    return customEntries;
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, CorrectSingleOptionCustomValueConfigTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::ValuesIn(getCustomEntries())));

const std::vector<std::string>& getPublicOptions() {
    static const std::vector<std::string> publicOptions = {
        KEY_LOG_LEVEL,
        VPU_CONFIG_KEY(LOG_LEVEL),
    };
    return publicOptions;
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, CorrectConfigPublicOptionsTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::ValuesIn(getPublicOptions())));

const std::vector<std::string>& getPrivateOptions() {
    static const std::vector<std::string> privateOptions = {
        InferenceEngine::MYRIAD_COPY_OPTIMIZATION,
    };
    return privateOptions;
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, CorrectConfigPrivateOptionsTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::ValuesIn(getPrivateOptions())));

const std::vector<std::map<std::string, std::string>>& getIncorrectConfigs() {
    static const std::vector<std::map<std::string, std::string>> incorrectConfigs = {
        {{KEY_LOG_LEVEL, "INCORRECT_LOG_LEVEL"}},

        {{InferenceEngine::MYRIAD_COPY_OPTIMIZATION, "ON"}},
        {{InferenceEngine::MYRIAD_COPY_OPTIMIZATION, "OFF"}},

        {{InferenceEngine::MYRIAD_PROTOCOL, "BLUETOOTH"}},
        {{InferenceEngine::MYRIAD_PROTOCOL, "LAN"}},

        {{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, "ON"}},
        {{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, "OFF"}},

        {{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, "ON"}},
        {{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, "OFF"}},

        {{InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "-10"}},

        {{InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, "ON"}},
        {{InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, "OFF"}},

        {{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "Two"}},
        {{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "SINGLE"}},

        {{InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, "ON"}},
        {{InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, "OFF"}},

        // Deprecated
        {{VPU_CONFIG_KEY(LOG_LEVEL), "INCORRECT_LOG_LEVEL"}},

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

        {
            {KEY_LOG_LEVEL, LOG_INFO},
            {InferenceEngine::MYRIAD_COPY_OPTIMIZATION, "ON"},
            {InferenceEngine::MYRIAD_PROTOCOL, "BLUETOOTH"},
            {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES)},
            {InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, "ON"},
            {InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB, "10"},
            {InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, "OFF"},
            {InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "1"},
            {InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS, "ON"},
        },
    };
    return incorrectConfigs;
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::ValuesIn(getIncorrectConfigs())),
    IncorrectConfigTests::getTestCaseName);

const std::vector<std::map<std::string, std::string>>& getIncorrectMultiConfigs() {
    static const std::vector<std::map<std::string, std::string>> incorrectMultiConfigs = {
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {KEY_LOG_LEVEL, "INCORRECT_LOG_LEVEL"},
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, "ON"},
        },

        // Deprecated
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_CONFIG_KEY(LOG_LEVEL), "INCORRECT_LOG_LEVEL"},
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), "ON"},
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_MYRIAD_CONFIG_KEY(PLATFORM), "-1"},
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_MYRIAD_CONFIG_KEY(PLATFORM), "0"},
        },
        {
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_MYRIAD_CONFIG_KEY(PLATFORM), "1"},
        },
    };
    return incorrectMultiConfigs;
}

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, IncorrectConfigTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MULTI),
        ::testing::ValuesIn(getIncorrectMultiConfigs())),
    IncorrectConfigTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigSingleOptionTests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::Values("INCORRECT_KEY")));

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, CorrectConfigAPITests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::Values(std::map<std::string, std::string>{})),
    CorrectConfigAPITests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, CorrectConfigAPITests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MULTI),
        ::testing::ValuesIn(getCorrectMultiConfigs())),
    CorrectConfigAPITests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigAPITests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::Values(std::map<std::string, std::string>{{"INCORRECT_KEY", "INCORRECT_VALUE"}})),
    IncorrectConfigAPITests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, IncorrectConfigAPITests,
    ::testing::Combine(
        ::testing::ValuesIn(getPrecisions()),
        ::testing::Values(CommonTestUtils::DEVICE_MULTI),
        ::testing::ValuesIn(getIncorrectMultiConfigs())),
    IncorrectConfigAPITests::getTestCaseName);

} // namespace
