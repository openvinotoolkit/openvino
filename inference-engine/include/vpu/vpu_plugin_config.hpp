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

#define VPU_CONFIG_KEY(name) InferenceEngine::VPUConfigParams::_CONFIG_KEY(VPU_##name)
#define VPU_CONFIG_VALUE(name) InferenceEngine::VPUConfigParams::VPU_##name

#define DECLARE_VPU_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPU_##name)
#define DECLARE_VPU_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(VPU_##name)

namespace InferenceEngine {
namespace VPUConfigParams {

/**
 * @brief Turn on HW stages usage (applicable for MyriadX devices only).
 * This option should be used with values: CONFIG_VALUE(YES) or CONFIG_VALUE(NO) (default)
 */
DECLARE_VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION);

/**
 * @brief The key to specify desirable log level for devices.
 * This option should be used with values: CONFIG_VALUE(LOG_NONE) (default),
 * CONFIG_VALUE(LOG_WARNING), CONFIG_VALUE(LOG_INFO), CONFIG_VALUE(LOG_DEBUG)
 */
DECLARE_VPU_CONFIG_KEY(LOG_LEVEL);

/**
 * @deprecated
 * @brief The key to define normalization coefficient for the network input.
 * This option should used with be a real number. Example "255.f"
 */
DECLARE_VPU_CONFIG_KEY(INPUT_NORM);

/**
 * @deprecated
 * @brief The flag to specify Bias value that is added to each element of the network input.
 * This option should used with be a real number. Example "0.1f"
 */
DECLARE_VPU_CONFIG_KEY(INPUT_BIAS);

/**
 * @brief The flag for adding to the profiling information the time of obtaining a tensor.
 * This option should be used with values: CONFIG_VALUE(YES) or CONFIG_VALUE(NO) (default)
 */
DECLARE_VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME);

/**
 * @brief The flag to reset stalled devices: CONFIG_VALUE(YES) or CONFIG_VALUE(NO) (default)
 * This is a plugin scope option and must be used with the plugin's SetConfig method
 */
DECLARE_VPU_CONFIG_KEY(FORCE_RESET);

/**
 * @brief This option allows to pass extra configuration for executable network.
 * By default, it is empty string, which means - no configuration.
 * String format:
 * <key>=<value>,<key>=<value>,...
 * Supported parameters and options:
 *   * file : path to XML file with configuration
 *   * data : options related to data objects (input, output, intermediate), next parameter describes the option
 *     * scale : SCALE factor for data range (applicable for input and intermediate data)
 */
DECLARE_VPU_CONFIG_KEY(NETWORK_CONFIG);

/**
 * @brief This option allows to to specify input output layouts for network layers.
 * By default, this value set to VPU_CONFIG_VALUE(AUTO) value.
 * Supported values:
 *   VPU_CONFIG_VALUE(AUTO) executable network configured to use optimal layer layout depending on available HW
 *   VPU_CONFIG_VALUE(NCHW) executable network forced to use NCHW input/output layouts
 *   VPU_CONFIG_VALUE(NHWC) executable network forced to use NHWC input/output layouts
 */
DECLARE_VPU_CONFIG_KEY(COMPUTE_LAYOUT);

/**
 * @brief This option allows to pass custom layers binding xml.
 * If layer is present in such an xml, it would be used during inference even if the layer is natively supported
 */
DECLARE_VPU_CONFIG_KEY(CUSTOM_LAYERS);

/**
 * @brief Supported keys definition for VPU_CONFIG_KEY(COMPUTE_LAYOUT) option.
 */
DECLARE_VPU_CONFIG_VALUE(AUTO);
DECLARE_VPU_CONFIG_VALUE(NCHW);
DECLARE_VPU_CONFIG_VALUE(NHWC);

/**
 * @brief This option allows to specify device.
 * If specified device is not available then creating infer request will throw an exception.
 */
DECLARE_VPU_CONFIG_KEY(PLATFORM);

/**
 * @brief Supported keys definition for VPU_CONFIG_KEY(PLATFORM) option.
 */
DECLARE_VPU_CONFIG_VALUE(2450);
DECLARE_VPU_CONFIG_VALUE(2480);

/**
 * @brief Ignore statistic in IR by plugin.
 * Plugin could use statistic present in IR in order to try to improve calculations precision.
 * If you don't want statistic to be used enable this option.
 * This option should be used with values: CONFIG_VALUE(YES) or CONFIG_VALUE(NO) (default)
 */
DECLARE_VPU_CONFIG_KEY(IGNORE_IR_STATISTIC);

}  // namespace VPUConfigParams
}  // namespace InferenceEngine
