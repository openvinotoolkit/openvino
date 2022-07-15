// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for properties that are passed from IE to plguinsor from one plugin to another
 * @file ie_internal_plugin_config.hpp
 */

#pragma once

#include "ie_plugin_config.hpp"
#include "openvino/runtime/properties.hpp"

namespace InferenceEngine {

/**
 * @brief A namespace with internal plugin configuration keys
 * @ingroup ie_dev_api_plugin_api
 */
namespace PluginConfigInternalParams {

/**
 * @def CONFIG_KEY_INTERNAL(name)
 * @ingroup ie_dev_api_plugin_api
 * @brief Shortcut for defining internal configuration keys
 */
#define CONFIG_KEY_INTERNAL(name) ::InferenceEngine::PluginConfigInternalParams::_CONFIG_KEY(name)

/**
 * @def CONFIG_VALUE_INTERNAL(name)
 * @ingroup ie_dev_api_plugin_api
 * @brief Shortcut for defining internal configuration values
 */
#define CONFIG_VALUE_INTERNAL(name) ::InferenceEngine::PluginConfigInternalParams::name

/**
 * @brief Defines a low precision mode key
 * @ingroup ie_dev_api_plugin_api
 */
DECLARE_CONFIG_KEY(LP_TRANSFORMS_MODE);

/**
 * @brief Limit \#threads that are used by CPU Executor Streams to execute `parallel_for` calls
 * @ingroup ie_dev_api_plugin_api
 */
DECLARE_CONFIG_KEY(CPU_THREADS_PER_STREAM);

/**
 * @brief Defines how many records can be stored in the CPU runtime parameters cache per CPU runtime parameter type per
 * stream
 * @ingroup ie_dev_api_plugin_api
 */
DECLARE_CONFIG_KEY(CPU_RUNTIME_CACHE_CAPACITY);

/**
 * @brief This key should be used to force disable export while loading network even if global cache dir is defined
 *        Used by HETERO plugin to disable automatic caching of subnetworks (set value to YES)
 * @ingroup ie_dev_api_plugin_api
 */
DECLARE_CONFIG_KEY(FORCE_DISABLE_CACHE);

/**
 * @brief Internal device id for particular device (like GPU.0, GPU.1 etc)
 */
DECLARE_CONFIG_KEY(CONFIG_DEVICE_ID);

}  // namespace PluginConfigInternalParams

}  // namespace InferenceEngine
