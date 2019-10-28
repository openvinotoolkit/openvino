// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header for advanced hardware related properties for clDNN plugin
 *        To use in SetConfig() method of plugins
 *
 * @file ie_plugin_config.hpp
 */
#pragma once

#include <string>
#include "ie_plugin_config.hpp"

namespace InferenceEngine {

/**
 * @brief GPU plugin configuration
 */
namespace CLDNNConfigParams {

/**
* @brief shortcut for defining configuration keys
*/
#define CLDNN_CONFIG_KEY(name) InferenceEngine::CLDNNConfigParams::_CONFIG_KEY(CLDNN_##name)
#define DECLARE_CLDNN_CONFIG_KEY(name) DECLARE_CONFIG_KEY(CLDNN_##name)
#define DECLARE_CLDNN_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(CLDNN_##name)

/**
* @brief This key instructs the clDNN plugin to use the OpenCL queue priority hint
* as defined in https://www.khronos.org/registry/OpenCL/specs/opencl-2.1-extensions.pdf
* this option should be used with an unsigned integer value (1 is lowest priority)
* 0 means no priority hint is set and default queue is created.
*/
DECLARE_CLDNN_CONFIG_KEY(PLUGIN_PRIORITY);

/**
* @brief This key instructs the clDNN plugin to use throttle hints the OpenCL queue throttle hint
* as defined in https://www.khronos.org/registry/OpenCL/specs/opencl-2.1-extensions.pdf,
* chapter 9.19. This option should be used with an unsigned integer value (1 is lowest energy consumption)
* 0 means no throttle hint is set and default queue created.
*/
DECLARE_CLDNN_CONFIG_KEY(PLUGIN_THROTTLE);

/**
* @brief This key controls clDNN memory pool optimization.
* Turned off by default.
*/
DECLARE_CLDNN_CONFIG_KEY(MEM_POOL);

/**
* @brief This key defines the directory name to which clDNN graph visualization will be dumped.
*/
DECLARE_CLDNN_CONFIG_KEY(GRAPH_DUMPS_DIR);

/**
* @brief This key defines the directory name to which full program sources will be dumped.
*/
DECLARE_CLDNN_CONFIG_KEY(SOURCES_DUMPS_DIR);

/**
* @brief This key turns usage of int8 optimizations and qunatized models on.
*/
DECLARE_CLDNN_CONFIG_KEY(INT8_ENABLED);

}  // namespace CLDNNConfigParams
}  // namespace InferenceEngine
