// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for properties that are passed from IE to plguinsor from one plugin to another
 * @file ie_internal_plugin_config.hpp
 */

#pragma once

#include "ie_plugin_config.hpp"

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
 * @brief This key should be used to force disable export while loading network even if global cache dir is defined
 *        Used by HETERO plugin to disable automatic caching of subnetworks (set value to YES)
 * @ingroup ie_dev_api_plugin_api
 */
DECLARE_CONFIG_KEY(FORCE_DISABLE_CACHE);

/**
 * @brief The name for setting work mode internal in MULTI device plugin option.
 *
 * This option should be used with value only:
 * PluginConfigInternalParams::MULTI_MODE_AUTO or PluginConfigInternalParams::MULTI_MODE_LEGACY
 */
DECLARE_CONFIG_KEY(WORK_MODE);
DECLARE_CONFIG_VALUE(MULTI_MODE_AUTO);

}  // namespace PluginConfigInternalParams

}  // namespace InferenceEngine
