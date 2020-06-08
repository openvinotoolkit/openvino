// Copyright (C) 2018-2019 Intel Corporation
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
    {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2450)}},
    {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2480)}},
    {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), ""}},
    // Deprecated
    {{VPU_CONFIG_KEY(PLATFORM), VPU_CONFIG_VALUE(2450)}},
    {{VPU_CONFIG_KEY(PLATFORM), VPU_CONFIG_VALUE(2480)}},
    {{VPU_CONFIG_KEY(PLATFORM), ""}}
};

static const std::vector<config_t> myriadIncorrectPlatformConfigValues = {
    {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "-1"}},
    {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), " 0"}},
    {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "MyriadX"}},
    // Deprecated
    {{VPU_CONFIG_KEY(PLATFORM), "-1"}},
    {{VPU_CONFIG_KEY(PLATFORM), " 0"}},
    {{VPU_CONFIG_KEY(PLATFORM), "MyriadX"}},
    // Deprecated key & value from current
    {{VPU_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2450)}},
    {{VPU_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2480)}},
    // Current key & deprecated value
    {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_CONFIG_VALUE(2450)}},
    {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_CONFIG_VALUE(2480)}},

};

static const std::vector<config_t> myriadCorrectProtocolConfigValues = {
    {{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), VPU_MYRIAD_CONFIG_VALUE(PCIE)}},
    {{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), VPU_MYRIAD_CONFIG_VALUE(USB)}},
    {{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), ""}},
};

static const std::vector<config_t> myriadIncorrectProtocolConfigValues = {
    // Protocols
    {{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), "0"}},
    {{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), "2450"}},
    {{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), "PCI"}},
};

static const std::vector<config_t> myriadCorrectConfigCombinationValues = {
    {{VPU_MYRIAD_CONFIG_KEY(PLATFORM), ""},
        {VPU_MYRIAD_CONFIG_KEY(PROTOCOL), ""}}
};

static const std::vector<config_t> myriadIncorrectPowerConfigValues = {
    {{VPU_MYRIAD_CONFIG_KEY(POWER_MANAGEMENT), "-1"}},
    {{VPU_MYRIAD_CONFIG_KEY(POWER_MANAGEMENT), "POWER_STANDARD"}},
    {{VPU_MYRIAD_CONFIG_KEY(POWER_MANAGEMENT), "INFER"}},
    {{VPU_MYRIAD_CONFIG_KEY(POWER_MANAGEMENT), ""}},
};

static const std::vector<config_t> myriadCorrectPowerConfigValues = {
    {{VPU_MYRIAD_CONFIG_KEY(POWER_MANAGEMENT), VPU_MYRIAD_CONFIG_VALUE(POWER_FULL)}},
    {{VPU_MYRIAD_CONFIG_KEY(POWER_MANAGEMENT), VPU_MYRIAD_CONFIG_VALUE(POWER_INFER)}},
    {{VPU_MYRIAD_CONFIG_KEY(POWER_MANAGEMENT), VPU_MYRIAD_CONFIG_VALUE(POWER_STAGE)}},
    {{VPU_MYRIAD_CONFIG_KEY(POWER_MANAGEMENT), VPU_MYRIAD_CONFIG_VALUE(POWER_STAGE_SHAVES)}},
    {{VPU_MYRIAD_CONFIG_KEY(POWER_MANAGEMENT), VPU_MYRIAD_CONFIG_VALUE(POWER_STAGE_NCES)}},
};

static const std::vector<config_t> myriadCorrectPackageTypeConfigValues = {
    {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(DDR_AUTO)}},
    {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(MICRON_2GB)}},
    {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(SAMSUNG_2GB)}},
    {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(HYNIX_2GB)}},
    {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(MICRON_1GB)}}
};

static const std::vector<config_t> myriadIncorrectPackageTypeConfigValues = {
    {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), "-1"}},
    {{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), "-MICRON_1GB"}},
};

IE_SUPPRESS_DEPRECATED_END

/// Platform
INSTANTIATE_TEST_CASE_P(MyriadPlatformConfigs, MyriadEngineSetCorrectConfigTest,
                        ::testing::ValuesIn(myriadCorrectPlatformConfigValues));

INSTANTIATE_TEST_CASE_P(MyriadPlatformConfigs, MyriadEngineSetIncorrectConfigTest,
                        ::testing::ValuesIn(myriadIncorrectPlatformConfigValues));

/// Protocol
INSTANTIATE_TEST_CASE_P(MyriadProtocolConfigs, MyriadEngineSetCorrectConfigTest,
                        ::testing::ValuesIn(myriadCorrectProtocolConfigValues));

INSTANTIATE_TEST_CASE_P(MyriadProtocolConfigs, MyriadEngineSetIncorrectConfigTest,
                        ::testing::ValuesIn(myriadIncorrectProtocolConfigValues));

/// Config combinations
INSTANTIATE_TEST_CASE_P(MyriadConfigOptionsCombination, MyriadEngineSetCorrectConfigTest,
                        ::testing::ValuesIn(myriadCorrectConfigCombinationValues));

/// Power Config
INSTANTIATE_TEST_CASE_P(MyriadPowerConfigs, MyriadEngineSetCorrectConfigTest,
                        ::testing::ValuesIn(myriadCorrectPowerConfigValues));

INSTANTIATE_TEST_CASE_P(MyriadPowerConfigs, MyriadEngineSetIncorrectConfigTest,
                        ::testing::ValuesIn(myriadIncorrectPowerConfigValues));
/// Package Config
INSTANTIATE_TEST_CASE_P(MyriadPackageConfigs, MyriadEngineSetCorrectConfigTest,
                        ::testing::ValuesIn(myriadCorrectPackageTypeConfigValues));

INSTANTIATE_TEST_CASE_P(MyriadPackageConfigs, MyriadEngineSetIncorrectConfigTest,
                        ::testing::ValuesIn(myriadIncorrectPackageTypeConfigValues));
