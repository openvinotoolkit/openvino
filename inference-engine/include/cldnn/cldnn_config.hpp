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

namespace InferenceEngine {

namespace Metrics {

/**
 * @def GPU_METRIC_KEY(name)
 * @brief shortcut for defining GPU plugin metrics
 */
#define GPU_METRIC_KEY(name) METRIC_KEY(GPU_##name)
#define DECLARE_GPU_METRIC_KEY(name, ...) DECLARE_METRIC_KEY(GPU_##name, __VA_ARGS__)

/**
 * @def DECLARE_GPU_METRIC_VALUE(name)
 * @brief shortcut for defining gpu metric values
 */
#define DECLARE_GPU_METRIC_VALUE(name) DECLARE_METRIC_VALUE(GPU_##name)

/**
 * @brief Metric which defines size of memory in bytes available for the device. For iGPU it returns host memory size, for dGPU - dedicated gpu memory size
 */
DECLARE_GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE, uint64_t);

/**
 * @brief Metric to get microarchitecture identifier in major.minor.revision format
 */
DECLARE_GPU_METRIC_KEY(UARCH_VERSION, std::string);

/**
 * @brief Metric to get count of execution units for current GPU
 */
DECLARE_GPU_METRIC_KEY(EXECUTION_UNITS_COUNT, int);

/**
 * @brief Possible return value for OPTIMIZATION_CAPABILITIES metric
 *  - "HW_MATMUL" - Defines if device has hardware block for matrix multiplication
 */
DECLARE_GPU_METRIC_VALUE(HW_MATMUL);

}  // namespace Metrics

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
DECLARE_CLDNN_CONFIG_KEY(NV12_TWO_INPUTS);

/**
 * @brief This key sets the max number of host threads that can be used by GPU plugin on model loading.
 * Default value is maximum number of threads available in the environment.
 */
DECLARE_CLDNN_CONFIG_KEY(MAX_NUM_THREADS);

/**
 * @brief Turning on this key enables to unroll recurrent layers such as TensorIterator or Loop with fixed iteration count.
 * This key is turned on by default. Turning this key on will achieve better inference performance for loops with not too many iteration counts (less than 16, as a rule of thumb).
 * Turning this key off will achieve better performance for both graph loading time and inference time with many iteration counts (greater than 16).
 * Note that turning this key on will increase the graph loading time in proportion to the iteration counts.
 * Thus, this key should be turned off if graph loading time is considered to be most important target to optimize.*/
DECLARE_CLDNN_CONFIG_KEY(ENABLE_LOOP_UNROLLING);

}  // namespace CLDNNConfigParams

namespace PluginConfigParams {

/**
 * @brief Optimize GPU plugin execution to maximize throughput.
 *
 * It is passed to Core::SetConfig(), this option should be used with values:
 * - KEY_GPU_THROUGHPUT_AUTO creates bare minimum of streams that might improve performance in some cases,
 *   this option allows to enable throttle hint for opencl queue thus reduce CPU load without significant performance
 * drop
 * - a positive integer value creates the requested number of streams
 */
DECLARE_CONFIG_VALUE(GPU_THROUGHPUT_AUTO);
DECLARE_CONFIG_KEY(GPU_THROUGHPUT_STREAMS);

/**
 * @brief This key enables dumping of the kernels used by the plugin for custom layers.
 *
 * This option should be used with values: PluginConfigParams::YES or PluginConfigParams::NO (default)
 */
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
DECLARE_CONFIG_KEY(TUNING_MODE);

DECLARE_CONFIG_VALUE(TUNING_CREATE);
DECLARE_CONFIG_VALUE(TUNING_USE_EXISTING);
DECLARE_CONFIG_VALUE(TUNING_DISABLED);
DECLARE_CONFIG_VALUE(TUNING_UPDATE);
DECLARE_CONFIG_VALUE(TUNING_RETUNE);

/**
 * @brief This key defines the tuning data filename to be created/used
 */
DECLARE_CONFIG_KEY(TUNING_FILE);

}  // namespace PluginConfigParams

}  // namespace InferenceEngine
