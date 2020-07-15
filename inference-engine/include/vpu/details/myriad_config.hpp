// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for VPU plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file myriad_config.hpp
 */

#pragma once

#include "common_utils.hpp"

#include "ie_plugin_config.hpp"
#include "ie_api.h"

#include <string>

namespace InferenceEngine {

/**
 * @brief The flag to reset stalled devices: CONFIG_VALUE(YES) or CONFIG_VALUE(NO) (default)
 * This is a plugin scope option and must be used with the plugin's SetConfig method
 */
DECLARE_MYRIAD_CONFIG_KEY(FORCE_RESET);

/**
 * @brief This option allows to specify device memory type.
 */
DECLARE_MYRIAD_CONFIG_KEY(DDR_TYPE);

/**
 * @brief Supported keys definition for CONFIG_KEY(MYRIAD_DDR_TYPE) option.
 */
DECLARE_MYRIAD_CONFIG_VALUE(AUTO);
DECLARE_MYRIAD_CONFIG_VALUE(MICRON_2GB);
DECLARE_MYRIAD_CONFIG_VALUE(SAMSUNG_2GB);
DECLARE_MYRIAD_CONFIG_VALUE(HYNIX_2GB);
DECLARE_MYRIAD_CONFIG_VALUE(MICRON_1GB);

/**
 * @brief This option allows to specify protocol.
 */
DECLARE_MYRIAD_CONFIG_KEY(PROTOCOL);

/**
 * @brief Supported keys definition for InferenceEngine::MYRIAD_PROTOCOL option.
 */
DECLARE_MYRIAD_CONFIG_VALUE(PCIE);
DECLARE_MYRIAD_CONFIG_VALUE(USB);

}  // namespace InferenceEngine
