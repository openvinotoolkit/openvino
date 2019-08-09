// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for VPU plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file vpu_plugin_config.hpp
 */

#pragma once

#include <string>

#include "ie_plugin_config.hpp"
#include "ie_api.h"

#define VPU_MYRIAD_CONFIG_KEY(name) InferenceEngine::VPUConfigParams::_CONFIG_KEY(VPU_MYRIAD_##name)
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

}  // namespace VPUConfigParams

}  // namespace InferenceEngine
