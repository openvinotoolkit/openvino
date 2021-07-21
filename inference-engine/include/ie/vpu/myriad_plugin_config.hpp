// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @deprecated Use vpu/myriad_config.hpp instead.
 * @brief A header that defines advanced related properties for VPU plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file myriad_plugin_config.hpp
 */

#pragma once

#include "ie_api.h"
#include "ie_plugin_config.hpp"

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
 * @deprecated Use InferenceEngine::MYRIAD_ENABLE_FORCE_RESET instead.
 * @brief The flag to reset stalled devices: CONFIG_VALUE(YES) or CONFIG_VALUE(NO) (default)
 * This is a plugin scope option and must be used with the plugin's SetConfig method
 */
INFERENCE_ENGINE_DEPRECATED("Use InferenceEngine::MYRIAD_ENABLE_FORCE_RESET instead")
DECLARE_VPU_MYRIAD_CONFIG_KEY(FORCE_RESET);

/**
 * @deprecated
 * @brief This option allows to specify device.
 * If specified device is not available then creating infer request will throw an exception.
 */
INFERENCE_ENGINE_DEPRECATED("")
DECLARE_VPU_MYRIAD_CONFIG_KEY(PLATFORM);

/**
 * @deprecated
 * @brief Supported keys definition for VPU_MYRIAD_CONFIG_KEY(PLATFORM) option.
 */
INFERENCE_ENGINE_DEPRECATED("")
DECLARE_VPU_MYRIAD_CONFIG_VALUE(2450);
INFERENCE_ENGINE_DEPRECATED("")
DECLARE_VPU_MYRIAD_CONFIG_VALUE(2480);

/**
 * @deprecated Use InferenceEngine::MYRIAD_DDR_TYPE instead
 * @brief This option allows to specify device memory type.
 */
INFERENCE_ENGINE_DEPRECATED("Use InferenceEngine::MYRIAD_DDR_TYPE instead")
DECLARE_VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE);

/**
 * @deprecated Use DDR type values from InferenceEngine namespace with MYRIAD_DDR_ prefix
 * @brief Supported keys definition for VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE) option.
 */
INFERENCE_ENGINE_DEPRECATED("Use InferenceEngine::MYRIAD_DDR_AUTO instead")
DECLARE_VPU_MYRIAD_CONFIG_VALUE(DDR_AUTO);
INFERENCE_ENGINE_DEPRECATED("Use InferenceEngine::MYRIAD_DDR_MICRON_2GB instead")
DECLARE_VPU_MYRIAD_CONFIG_VALUE(MICRON_2GB);
INFERENCE_ENGINE_DEPRECATED("Use InferenceEngine::MYRIAD_DDR_SAMSUNG_2GB instead")
DECLARE_VPU_MYRIAD_CONFIG_VALUE(SAMSUNG_2GB);
INFERENCE_ENGINE_DEPRECATED("Use InferenceEngine::MYRIAD_DDR_HYNIX_2GB instead")
DECLARE_VPU_MYRIAD_CONFIG_VALUE(HYNIX_2GB);
INFERENCE_ENGINE_DEPRECATED("Use InferenceEngine::MYRIAD_DDR_MICRON_1GB instead")
DECLARE_VPU_MYRIAD_CONFIG_VALUE(MICRON_1GB);

}  // namespace VPUConfigParams

}  // namespace InferenceEngine
