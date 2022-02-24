// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "myriad_test_case.h"
#include "vpu/myriad_config.hpp"
#include "vpu/private_plugin_config.hpp"

using MyriadEngineSetCorrectConfigTest = MyriadEngineSetConfigTest;
using MyriadEngineSetIncorrectConfigTest = MyriadEngineSetConfigTest;

TEST_P(MyriadEngineSetCorrectConfigTest, SetCorrectConfig) {
    ASSERT_NO_THROW(myriad_engine_->SetConfig(GetParam()));
}

TEST_P(MyriadEngineSetIncorrectConfigTest, SetIncorrectConfig) {
    ASSERT_ANY_THROW(myriad_engine_->SetConfig(GetParam()));
}

static const std::vector<config_t> myriadCorrectProtocolConfigValues = {
    {{InferenceEngine::MYRIAD_PROTOCOL, InferenceEngine::MYRIAD_PCIE}},
    {{InferenceEngine::MYRIAD_PROTOCOL, InferenceEngine::MYRIAD_USB}},
    {{InferenceEngine::MYRIAD_PROTOCOL, ""}},
};

static const std::vector<config_t> myriadIncorrectProtocolConfigValues = {
    // Protocols
    {{InferenceEngine::MYRIAD_PROTOCOL, "0"}},
    {{InferenceEngine::MYRIAD_PROTOCOL, "PCI"}},
};

static const std::vector<config_t> myriadCorrectConfigCombinationValues = {
    {{InferenceEngine::MYRIAD_PROTOCOL, ""}}
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
};

static const std::vector<config_t> myriadIncorrectPackageTypeConfigValues = {
    {{InferenceEngine::MYRIAD_DDR_TYPE, "-1"}},
    {{InferenceEngine::MYRIAD_DDR_TYPE, "-MICRON_1GB"}},
};

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
