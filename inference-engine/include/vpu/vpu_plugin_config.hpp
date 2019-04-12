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

#define VPU_HDDL_CONFIG_KEY(name) InferenceEngine::VPUConfigParams::_CONFIG_KEY(VPU_HDDL_##name)
#define VPU_HDDL_CONFIG_VALUE(name) InferenceEngine::VPUConfigParams::VPU_HDDL_##name

#define DECLARE_VPU_HDDL_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPU_HDDL_##name)
#define DECLARE_VPU_HDDL_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(VPU_HDDL_##name)

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
 * @brief [Only for HDDLPlugin]
 * Type: Arbitrary non-empty string. If empty (""), equals no set, default: "";
 * This option allows to specify the number of MYX devices used for inference a specific Executable network.
 * Note: Only one network would be allocated to one device.
 * The number of devices for the tag is specified in the hddl_service.config file.
 * Example:
 * "service_settings":
 * {
 *     "graph_tag_map":
 *     {
 *         "tagA":3
 *     }
 * }
 * It means that an executable network marked with tagA will be executed on 3 devices
 */
DECLARE_VPU_HDDL_CONFIG_KEY(GRAPH_TAG);

/**
 * @brief [Only for HDDLPlugin]
 * Type: Arbitrary non-empty string. If empty (""), equals no set, default: "";
 * This config makes the executable networks to be allocated on one certain device (instead of multiple devices).
 * And all inference through this executable network, will be done on this device.
 * Note: Only one network would be allocated to one device.
 * The number of devices which will be used for stream-affinity must be specified in hddl_service.config file.
 * Example:
 * "service_settings":
 * {
 *     "stream_device_number":5
 * }
 * It means that 5 device will be used for stream-affinity
 */
DECLARE_VPU_HDDL_CONFIG_KEY(STREAM_ID);

/**
 * @brief [Only for HDDLPlugin]
 * Type: Arbitrary non-empty string. If empty (""), equals no set, default: "";
 * This config allows user to control device flexibly. This config gives a "tag" for a certain device while
 * allocating a network to it. Afterward, user can allocating/deallocating networks to this device with this "tag".
 * Devices used for such use case is controlled by a so-called "Bypass Scheduler" in HDDL backend, and the number
 * of such device need to be specified in hddl_service.config file.
 * Example:
 * "service_settings":
 * {
 *     "bypass_device_number": 5
 * }
 * It means that 5 device will be used for Bypass scheduler.
 */
DECLARE_VPU_HDDL_CONFIG_KEY(DEVICE_TAG);

/**
 * @brief [Only for HDDLPlugin]
 * Type: "YES/NO", default is "NO".
 * This config is a sub-config of DEVICE_TAG, and only available when "DEVICE_TAG" is set. After a user load a
 * network, the user got a handle for the network.
 * If "YES", the network allocated is bind to the device (with the specified "DEVICE_TAG"), which means all afterwards
 * inference through this network handle will be executed on this device only.
 * If "NO", the network allocated is not bind to the device (with the specified "DEVICE_TAG"). If the same network
 * is allocated on multiple other devices (also set BIND_DEVICE to "False"), then inference through any handle of these
 * networks may be executed on any of these devices those have the network loaded.
 */
DECLARE_VPU_HDDL_CONFIG_KEY(BIND_DEVICE);

/**
 * @brief [Only for HDDLPlugin]
 * Type: A signed int wrapped in a string, default is "0".
 * This config is a sub-config of DEVICE_TAG, and only available when "DEVICE_TAG" is set and "BIND_DEVICE" is "False".
 * When there are multiple devices running a certain network (a same network running on multiple devices in Bypass Scheduler),
 * the device with a larger number has a higher priority, and more inference tasks will be fed to it with priority.
 */
DECLARE_VPU_HDDL_CONFIG_KEY(RUNTIME_PRIORITY);


/**
 * @brief [Only for HDDLPlugin]
 * Type: "YES/NO", default is "NO". **Note: ONLY available when "DEVICE_TAG" is set.
 * This config should be used only when the network has been loaded already with the same network content, the same
 * "DEVICE_TAG" as used this time and "BIND_DEVICE" of the loaded network had been set to "NO".
 * This config is only used to update the "RUNTIME_PRIORITY" of previous loaded network, and the application should keep using
 * the network handle that previous allocated to do inference.
 * - If "Yes": the "RUNTIME_PRIORITY" must be specified with a integer, and it will be set as the new runtime priority for that network on that device.
 * - If "No": load this network to deivce.
 * **Note: If "BIND_DEVICE" of the previously loaded network was "Yes", the behavior of "update runtime priority" is undefined.
 */
DECLARE_VPU_HDDL_CONFIG_KEY(UPDATE_RUNTIME_PRIORITY);

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
