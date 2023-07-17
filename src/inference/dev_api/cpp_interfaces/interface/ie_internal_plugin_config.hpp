// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for properties that are passed from IE to plguinsor from one plugin to another
 * @file ie_internal_plugin_config.hpp
 */

#pragma once

#include "ie_plugin_config.hpp"
#include "openvino/runtime/internal_properties.hpp"

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
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_CONFIG_KEY(LP_TRANSFORMS_MODE);

/**
 * @brief Limit \#threads that are used by CPU Executor Streams to execute `parallel_for` calls
 * @ingroup ie_dev_api_plugin_api
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_CONFIG_KEY(CPU_THREADS_PER_STREAM);

/**
 * @brief Number of streams in Performance-core(big core)
 * @ingroup ie_dev_api_plugin_api
 * @brief Shortcut for defining internal configuration values
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_CONFIG_KEY(BIG_CORE_STREAMS);

/**
 * @brief Number of streams in Efficient-core(small core) on hybrid cores machine
 * @ingroup ie_dev_api_plugin_api
 * @brief Shortcut for defining internal configuration values
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_CONFIG_KEY(SMALL_CORE_STREAMS);

/**
 * @brief Number of threads per stream in big cores
 * @ingroup ie_dev_api_plugin_api
 * @brief Shortcut for defining internal configuration values
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_CONFIG_KEY(THREADS_PER_STREAM_BIG);

/**
 * @brief Number of threads per stream in small cores on hybrid cores machine
 * @ingroup ie_dev_api_plugin_api
 * @brief Shortcut for defining internal configuration values
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_CONFIG_KEY(THREADS_PER_STREAM_SMALL);

/**
 * @brief Small core start offset when binding cpu cores
 * @ingroup ie_dev_api_plugin_api
 * @brief Shortcut for defining internal configuration values
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_CONFIG_KEY(SMALL_CORE_OFFSET);

/**
 * @brief Defines how many records can be stored in the CPU runtime parameters cache per CPU runtime parameter type per
 * stream
 * @ingroup ie_dev_api_plugin_api
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_CONFIG_KEY(CPU_RUNTIME_CACHE_CAPACITY);

/**
 * @brief Internal device id for particular device (like GPU.0, GPU.1 etc)
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_CONFIG_KEY(CONFIG_DEVICE_ID);

/**
 * @brief enable hyper thread
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_CONFIG_KEY(ENABLE_HYPER_THREAD);

/**
 * @brief Defines Snippets tokenization mode
 *      @param ENABLE - default pipeline
 *      @param IGNORE_CALLBACK - disable the Snippets markup transformation and tokenization callback
 *      @param DISABLE - turn off the Snippets
 * @ingroup ie_dev_api_plugin_api
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_CONFIG_KEY(SNIPPETS_MODE);
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_CONFIG_VALUE(ENABLE);
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_CONFIG_VALUE(IGNORE_CALLBACK);
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_CONFIG_VALUE(DISABLE);

}  // namespace PluginConfigInternalParams

}  // namespace InferenceEngine
