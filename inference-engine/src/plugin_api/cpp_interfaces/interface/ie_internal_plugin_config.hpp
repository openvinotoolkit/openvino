// Copyright (C) 2018-2020 Intel Corporation
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
#define CONFIG_KEY_INTERNAL(name)  ::InferenceEngine::PluginConfigInternalParams::_CONFIG_KEY(name)

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
 * @brief This key should be used to notify aggregating plugin
 *        that it is used inside other aggregating plugin
 * @ingroup ie_dev_api_plugin_api
 */
DECLARE_CONFIG_KEY(AGGREGATED_PLUGIN);

}  // namespace PluginConfigInternalParams

}  // namespace InferenceEngine
