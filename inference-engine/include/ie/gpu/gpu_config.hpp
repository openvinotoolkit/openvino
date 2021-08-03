// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware related properties for GPU plugin
 *        To use in SetConfig() method of plugins
 *
 * @file gpu_config.hpp
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
namespace GPUConfigParams {

/**
 * @brief shortcut for defining configuration keys
 */
#define GPU_CONFIG_KEY(name) InferenceEngine::GPUConfigParams::_CONFIG_KEY(GPU_##name)
#define DECLARE_GPU_CONFIG_KEY(name) DECLARE_CONFIG_KEY(GPU_##name)
#define DECLARE_GPU_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(GPU_##name)

/**
 * @brief This key instructs the GPU plugin to use the OpenCL queue priority hint
 * as defined in https://www.khronos.org/registry/OpenCL/specs/opencl-2.1-extensions.pdf
 * this option should be used with an unsigned integer value (1 is lowest priority)
 * 0 means no priority hint is set and default queue is created.
 */
DECLARE_GPU_CONFIG_KEY(PLUGIN_PRIORITY);

/**
 * @brief This key instructs the GPU plugin to use throttle hints the OpenCL queue throttle hint
 * as defined in https://www.khronos.org/registry/OpenCL/specs/opencl-2.1-extensions.pdf,
 * chapter 9.19. This option should be used with an unsigned integer value (1 is lowest energy consumption)
 * 0 means no throttle hint is set and default queue created.
 */
DECLARE_GPU_CONFIG_KEY(PLUGIN_THROTTLE);

/**
 * @brief This key should be set to correctly handle NV12 input without pre-processing.
 * Turned off by default.
 */
DECLARE_GPU_CONFIG_KEY(NV12_TWO_INPUTS);

/**
 * @brief This key sets the max number of host threads that can be used by GPU plugin on model loading.
 * Default value is maximum number of threads available in the environment.
 */
DECLARE_GPU_CONFIG_KEY(MAX_NUM_THREADS);

/**
 * @brief Turning on this key enables to unroll recurrent layers such as TensorIterator or Loop with fixed iteration count.
 * This key is turned on by default. Turning this key on will achieve better inference performance for loops with not too many iteration counts (less than 16, as a rule of thumb).
 * Turning this key off will achieve better performance for both graph loading time and inference time with many iteration counts (greater than 16).
 * Note that turning this key on will increase the graph loading time in proportion to the iteration counts.
 * Thus, this key should be turned off if graph loading time is considered to be most important target to optimize.*/
DECLARE_GPU_CONFIG_KEY(ENABLE_LOOP_UNROLLING);

}  // namespace GPUConfigParams

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
}  // namespace PluginConfigParams

}  // namespace InferenceEngine
