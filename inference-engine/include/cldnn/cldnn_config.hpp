// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware related properties for clDNN plugin
 *        To use in SetConfig() method of plugins
 *
 * @file cldnn_config.hpp
 */
#pragma once

#include "ie_plugin_config.hpp"
#include "ie_api.h"
#include "gpu/gpu_config.hpp"

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
INFERENCE_ENGINE_DEPRECATED("Use InferenceEngine::GPUConfigParams::GPU_PLUGIN_PRIORITY instead")
DECLARE_CLDNN_CONFIG_KEY(PLUGIN_PRIORITY);

/**
 * @brief This key instructs the clDNN plugin to use throttle hints the OpenCL queue throttle hint
 * as defined in https://www.khronos.org/registry/OpenCL/specs/opencl-2.1-extensions.pdf,
 * chapter 9.19. This option should be used with an unsigned integer value (1 is lowest energy consumption)
 * 0 means no throttle hint is set and default queue created.
 */
INFERENCE_ENGINE_DEPRECATED("Use InferenceEngine::GPUConfigParams::GPU_PLUGIN_THROTTLE instead")
DECLARE_CLDNN_CONFIG_KEY(PLUGIN_THROTTLE);

/**
 * @brief This key controls clDNN memory pool optimization.
 * Turned off by default.
 */
INFERENCE_ENGINE_DEPRECATED("The config key will be removed")
DECLARE_CLDNN_CONFIG_KEY(MEM_POOL);

/**
 * @brief This key defines the directory name to which clDNN graph visualization will be dumped.
 */
INFERENCE_ENGINE_DEPRECATED("The config key will be removed")
DECLARE_CLDNN_CONFIG_KEY(GRAPH_DUMPS_DIR);

/**
 * @brief This key defines the directory name to which full program sources will be dumped.
 */
INFERENCE_ENGINE_DEPRECATED("The config key will be removed")
DECLARE_CLDNN_CONFIG_KEY(SOURCES_DUMPS_DIR);

/**
 * @brief This key enables FP16 precision for quantized models.
 * By default the model is converted to FP32 precision before running LPT. If this key is enabled (default), then non-quantized layers
 * will be converted back to FP16 after LPT, which might imrpove the performance if a model has a lot of compute operations in
 * non-quantized path. This key has no effect if current device doesn't have INT8 optimization capabilities.
 */
DECLARE_CLDNN_CONFIG_KEY(ENABLE_FP16_FOR_QUANTIZED_MODELS);

/**
 * @brief This key should be set to correctly handle NV12 input without pre-processing.
 * Turned off by default.
 */
INFERENCE_ENGINE_DEPRECATED("Use InferenceEngine::GPUConfigParams::GPU_NV12_TWO_INPUTS instead")
DECLARE_CLDNN_CONFIG_KEY(NV12_TWO_INPUTS);

}  // namespace CLDNNConfigParams

namespace PluginConfigParams {

/**
 * @brief This key enables dumping of the kernels used by the plugin for custom layers.
 *
 * This option should be used with values: PluginConfigParams::YES or PluginConfigParams::NO (default)
 */
INFERENCE_ENGINE_DEPRECATED("The config key will be removed")
DECLARE_CONFIG_KEY(DUMP_KERNELS);

/**
 * @brief This key controls performance tuning done or used by the plugin.
 *
 * This option should be used with values:
 * PluginConfigParams::TUNING_DISABLED (default)
 * PluginConfigParams::TUNING_USE_EXISTING - use existing data from tuning file
 * PluginConfigParams::TUNING_CREATE - create tuning data for parameters not present in tuning file
 * PluginConfigParams::TUNING_UPDATE - perform non-tuning updates like removal of invalid/deprecated data
 * PluginConfigParams::TUNING_RETUNE - create tuning data for all parameters, even if already present
 *
 * For values TUNING_CREATE and TUNING_RETUNE the file will be created if it does not exist.
 */
INFERENCE_ENGINE_DEPRECATED("The config key will be removed")
DECLARE_CONFIG_KEY(TUNING_MODE);

INFERENCE_ENGINE_DEPRECATED("The config value will be removed")
DECLARE_CONFIG_VALUE(TUNING_CREATE);
INFERENCE_ENGINE_DEPRECATED("The config value will be removed")
DECLARE_CONFIG_VALUE(TUNING_USE_EXISTING);
INFERENCE_ENGINE_DEPRECATED("The config value will be removed")
DECLARE_CONFIG_VALUE(TUNING_DISABLED);
INFERENCE_ENGINE_DEPRECATED("The config value will be removed")
DECLARE_CONFIG_VALUE(TUNING_UPDATE);
INFERENCE_ENGINE_DEPRECATED("The config value will be removed")
DECLARE_CONFIG_VALUE(TUNING_RETUNE);

/**
 * @brief This key defines the tuning data filename to be created/used
 */
INFERENCE_ENGINE_DEPRECATED("The config key will be removed")
DECLARE_CONFIG_KEY(TUNING_FILE);

}  // namespace PluginConfigParams

}  // namespace InferenceEngine
