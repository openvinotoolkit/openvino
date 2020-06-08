// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for VPU plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file
 */

#pragma once

#include <string>

#include "ie_plugin_config.hpp"
#include "ie_api.h"

/**
 * @def VPU_MYRIAD_CONFIG_KEY(name)
 * @brief Shortcut for defining VPU MYRIAD configuration key
 */
#define VPU_MYRIAD_CONFIG_KEY(name) InferenceEngine::VPUConfigParams::_CONFIG_KEY(VPU_MYRIAD_##name)
/**
 * @def VPU_MYRIAD_CONFIG_VALUE(name)
 * @brief Shortcut for defining VPU MYRIAD configuration value
 */
#define VPU_MYRIAD_CONFIG_VALUE(name) InferenceEngine::VPUConfigParams::VPU_MYRIAD_##name

#define DECLARE_VPU_MYRIAD_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPU_MYRIAD_##name)
#define DECLARE_VPU_MYRIAD_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(VPU_MYRIAD_##name)

namespace InferenceEngine {

namespace VPUConfigParams {

/**
 * @brief The flag to reset stalled devices: CONFIG_VALUE(YES) or CONFIG_VALUE(NO) (default)
 * This is a plugin scope option and must be used with the plugin's SetConfig method
 */
DECLARE_VPU_MYRIAD_CONFIG_KEY(FORCE_RESET);

/**
 * @brief This option allows to specify device.
 * If specified device is not available then creating infer request will throw an exception.
 */
DECLARE_VPU_MYRIAD_CONFIG_KEY(PLATFORM);

/**
 * @brief Supported keys definition for VPU_MYRIAD_CONFIG_KEY(PLATFORM) option.
 */
DECLARE_VPU_MYRIAD_CONFIG_VALUE(2450);
DECLARE_VPU_MYRIAD_CONFIG_VALUE(2480);

/**
 * @brief This option allows to specify device memory type.
 */
DECLARE_VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE);

/**
 * @brief Supported keys definition for VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE) option.
 */
DECLARE_VPU_MYRIAD_CONFIG_VALUE(DDR_AUTO);
DECLARE_VPU_MYRIAD_CONFIG_VALUE(MICRON_2GB);
DECLARE_VPU_MYRIAD_CONFIG_VALUE(SAMSUNG_2GB);
DECLARE_VPU_MYRIAD_CONFIG_VALUE(HYNIX_2GB);
DECLARE_VPU_MYRIAD_CONFIG_VALUE(MICRON_1GB);

}  // namespace VPUConfigParams

}  // namespace InferenceEngine
