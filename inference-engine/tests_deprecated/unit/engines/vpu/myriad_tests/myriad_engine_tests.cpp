// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "myriad_test_case.h"

using MyriadEngineSetCorrectConfigTest = MyriadEngineSetConfigTest;
using MyriadEngineSetIncorrectConfigTest = MyriadEngineSetConfigTest;

TEST_P(MyriadEngineSetCorrectConfigTest, SetCorrectConfig) {
    ASSERT_NO_THROW(myriad_engine_->SetConfig(GetParam()));
}

TEST_P(MyriadEngineSetIncorrectConfigTest, SetIncorrectConfig) {
    ASSERT_ANY_THROW(myriad_engine_->SetConfig(GetParam()));
}

IE_SUPPRESS_DEPRECATED_START

static const std::vector<config_t> myriadCorrectPlatformConfigValues = {
    // Deprecated
    {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2450)}},
    {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2480)}},
    {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), ""}}
};

static const std::vector<config_t> myriadIncorrectPlatformConfigValues = {
    // Deprecated
    {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "-1"}},
    {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), " 0"}},
    {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "MyriadX"}}
};

static const std::vector<config_t> myriadCorrectProtocolConfigValues = {
    {{InferenceEngine::MYRIAD_PROTOCOL, InferenceEngine::MYRIAD_PCIE}},
    {{InferenceEngine::MYRIAD_PROTOCOL, InferenceEngine::MYRIAD_USB}},
    {{InferenceEngine::MYRIAD_PROTOCOL, ""}},

    // Deprecated
    {{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), VPU_MYRIAD_CONFIG_VALUE(PCIE)}},
    {{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), VPU_MYRIAD_CONFIG_VALUE(USB)}},
    {{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), ""}},
};

static const std::vector<config_t> myriadIncorrectProtocolConfigValues = {
    // Protocols
    {{InferenceEngine::MYRIAD_PROTOCOL, "0"}},
    {{InferenceEngine::MYRIAD_PROTOCOL, "2450"}},
    {{InferenceEngine::MYRIAD_PROTOCOL, "PCI"}},

    // Deprecated
    {{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), "0"}},
    {{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), "2450"}},
    {{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), "PCI"}},
};

static const std::vector<config_t> myriadCorrectConfigCombinationValues = {
    {{InferenceEngine::MYRIAD_PROTOCOL, ""},
    // Deprecated
    {VPU_MYRIAD_CONFIG_KEY(PROTOCOL), ""}}
};

static const std::vector<config_t> myriadIncorrectPowerConfigValues = {
    {{InferenceEngine::MYRIAD_POWER_MANAGEMENT, "-1"}},
    {{InferenceEngine::MYRIAD_POWER_MANAGEMENT, "POWER_STANDARD"}},
    {{InferenceEngine::MYRIAD_POWER_MANAGEMENT, "INFER"}},
    {{InferenceEngine::MYRIAD_POWER_MANAGEMENT, ""}},
};

static const std::vector<config_t> myriadCorrectPowerConfigValues = {
    {{InferenceEngine::MYRIAD_POWER_MANAGEMENT, InferenceEngine::MYRIAD_POWER_FULL}},
    {{InferenceEngine::MYRIAD_POWER_MANAGEMENT, InferenceEngine::MYRIAD_POWER_INFER}},
    {{InferenceEngine::MYRIAD_POWER_MANAGEMENT, InferenceEngine::MYRIAD_POWER_STAGE}},
    {{InferenceEngine::MYRIAD_POWER_MANAGEMENT, InferenceEngine::MYRIAD_POWER_STAGE_SHAVES}},
    {{InferenceEngine::MYRIAD_POWER_MANAGEMENT, InferenceEngine::MYRIAD_POWER_STAGE_NCES}},
};

static const std::vector<config_t> myriadCorrectPackageTypeConfigValues = {
    {{InferenceEngine::MYRIAD_DDR_TYPE, InferenceEngine::MYRIAD_DDR_AUTO}},
    {{InferenceEngine::MYRIAD_DDR_TYPE, InferenceEngine::MYRIAD_DDR_MICRON_2GB}},
    {{InferenceEngine::MYRIAD_DDR_TYPE, InferenceEngine::MYRIAD_DDR_SAMSUNG_2GB}},
    {{InferenceEngine::MYRIAD_DDR_TYPE, InferenceEngine::MYRIAD_DDR_HYNIX_2GB}},
    {{InferenceEngine::MYRIAD_DDR_TYPE, InferenceEngine::MYRIAD_DDR_MICRON_1GB}},

    // Deprecated
    {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(DDR_AUTO)}},
    {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(MICRON_2GB)}},
    {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(SAMSUNG_2GB)}},
    {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(HYNIX_2GB)}},
    {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(MICRON_1GB)}},
};

static const std::vector<config_t> myriadIncorrectPackageTypeConfigValues = {
    {{InferenceEngine::MYRIAD_DDR_TYPE, "-1"}},
    {{InferenceEngine::MYRIAD_DDR_TYPE, "-MICRON_1GB"}},

    // Deprecated
    {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), "-1"}},
    {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), "-MICRON_1GB"}},
};

IE_SUPPRESS_DEPRECATED_END

/// Platform
INSTANTIATE_TEST_SUITE_P(MyriadPlatformConfigs, MyriadEngineSetCorrectConfigTest,
                        ::testing::ValuesIn(myriadCorrectPlatformConfigValues));

INSTANTIATE_TEST_SUITE_P(MyriadPlatformConfigs, MyriadEngineSetIncorrectConfigTest,
                        ::testing::ValuesIn(myriadIncorrectPlatformConfigValues));

/// Protocol
INSTANTIATE_TEST_SUITE_P(MyriadProtocolConfigs, MyriadEngineSetCorrectConfigTest,
                        ::testing::ValuesIn(myriadCorrectProtocolConfigValues));

INSTANTIATE_TEST_SUITE_P(MyriadProtocolConfigs, MyriadEngineSetIncorrectConfigTest,
                        ::testing::ValuesIn(myriadIncorrectProtocolConfigValues));

/// Config combinations
INSTANTIATE_TEST_SUITE_P(MyriadConfigOptionsCombination, MyriadEngineSetCorrectConfigTest,
                        ::testing::ValuesIn(myriadCorrectConfigCombinationValues));

/// Power Config
INSTANTIATE_TEST_SUITE_P(MyriadPowerConfigs, MyriadEngineSetCorrectConfigTest,
                        ::testing::ValuesIn(myriadCorrectPowerConfigValues));

INSTANTIATE_TEST_SUITE_P(MyriadPowerConfigs, MyriadEngineSetIncorrectConfigTest,
                        ::testing::ValuesIn(myriadIncorrectPowerConfigValues));
/// Package Config
INSTANTIATE_TEST_SUITE_P(MyriadPackageConfigs, MyriadEngineSetCorrectConfigTest,
                        ::testing::ValuesIn(myriadCorrectPackageTypeConfigValues));

INSTANTIATE_TEST_SUITE_P(MyriadPackageConfigs, MyriadEngineSetIncorrectConfigTest,
                        ::testing::ValuesIn(myriadIncorrectPackageTypeConfigValues));
