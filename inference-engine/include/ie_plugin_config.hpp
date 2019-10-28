// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header for advanced hardware related properties for IE plugins
 *        To use in SetConfig() method of plugins
 *        LoadNetwork() method overloads that accept config as parameter
 *
 * @file ie_plugin_config.hpp
 */
#pragma once

#include <string>
#include <tuple>
#include <vector>

namespace InferenceEngine {

/**
 * @brief %Metrics
 */
namespace Metrics {

#ifndef DECLARE_METRIC_KEY_IMPL
# define DECLARE_METRIC_KEY_IMPL(...)
#endif

/**
 * @def METRIC_KEY(name)
 * @brief shortcut for defining common Inference Engine metrics
 */
#define METRIC_KEY(name) InferenceEngine::Metrics::METRIC_##name

/**
 * @def EXEC_NETWORK_METRIC_KEY(name)
 * @brief shortcut for defining common Inference Engine ExecutableNetwork metrics
 */
#define EXEC_NETWORK_METRIC_KEY(name) METRIC_KEY(name)

#define DECLARE_METRIC_KEY(name, ...)               \
    static constexpr auto METRIC_##name = #name;    \
    DECLARE_METRIC_KEY_IMPL(name, __VA_ARGS__)

#define DECLARE_EXEC_NETWORK_METRIC_KEY(name, ...) DECLARE_METRIC_KEY(name, __VA_ARGS__)

/**
 * @def METRIC_VALUE(name)
 * @brief shortcut for defining metric values
 */
#define METRIC_VALUE(name) InferenceEngine::Metrics::name
#define DECLARE_METRIC_VALUE(name) static constexpr auto name = #name

/**
* @brief Metric to get a std::vector<std::string> of available device IDs. String value is "AVAILABLE_DEVICES"
*/
DECLARE_METRIC_KEY(AVAILABLE_DEVICES, std::vector<std::string>);

/**
* @brief Metric to get a std::vector<std::string> of supported metrics. String value is "SUPPORTED_METRICS"
* This can be used as an executable network metric as well.
*
* Each of the returned device metrics can be passed to Core::GetMetric, executable network metrics
* can be passed to ExecutableNetwork::GetMetric.
*
*/
DECLARE_METRIC_KEY(SUPPORTED_METRICS, std::vector<std::string>);

/**
* @brief Metric to get a std::vector<std::string> of supported config keys. String value is "SUPPORTED_CONFIG_KEYS"
* This can be used as an executable network metric as well.
*
* Each of the returned device configuration keys can be passed to Core::SetConfig, Core::GetConfig, and Core::LoadNetwork,
* configuration keys for executable networks can be passed to ExecutableNetwork::SetConfig and ExecutableNetwork::GetConfig.
*
*/
DECLARE_METRIC_KEY(SUPPORTED_CONFIG_KEYS, std::vector<std::string>);

/**
* @brief Metric to get a std::string value representing a full device name. String value is "FULL_DEVICE_NAME"
*/
DECLARE_METRIC_KEY(FULL_DEVICE_NAME, std::string);

/**
* @brief Metric to get a std::vector<std::string> of optimization options per device. String value is "OPTIMIZATION_CAPABILITIES"
* The possible values:
*  - "FP32" - device can support FP32 models
*  - "FP16" - device can support FP16 models
*  - "INT8" - device can support models with INT8 layers
*  - "BIN" - device can support models with BIN layers
*  - "WINOGRAD" - device can support models where convolution implemented via Winograd transformations
*/
DECLARE_METRIC_KEY(OPTIMIZATION_CAPABILITIES, std::vector<std::string>);

DECLARE_METRIC_VALUE(FP32);
DECLARE_METRIC_VALUE(FP16);
DECLARE_METRIC_VALUE(INT8);
DECLARE_METRIC_VALUE(BIN);
DECLARE_METRIC_VALUE(WINOGRAD);

/**
* @brief Metric to provide information about a range for streams on platforms where streams are supported.
* Metric returns a value of std::tuple<unsigned int, unsigned int> type, where:
*  - First value is bottom bound.
*  - Second value is upper bound.
* String value for metric name is "RANGE_FOR_STREAMS".
*/
DECLARE_METRIC_KEY(RANGE_FOR_STREAMS, std::tuple<unsigned int, unsigned int>);

/**
* @brief Metric to provide a hint for a range for number of async infer requests. If device supports streams,
* the metric provides range for number of IRs per stream.
* Metric returns a value of std::tuple<unsigned int, unsigned int, unsigned int> type, where:
*  - First value is bottom bound.
*  - Second value is upper bound.
*  - Third value is step inside this range.
* String value for metric name is "RANGE_FOR_ASYNC_INFER_REQUESTS".
*/
DECLARE_METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS, std::tuple<unsigned int, unsigned int, unsigned int>);

/**
* @brief Metric to get an unsigned int value of number of waiting infer request.
* String value is "NUMBER_OF_WAITNING_INFER_REQUESTS". This can be used as an executable network metric as well
*/
DECLARE_METRIC_KEY(NUMBER_OF_WAITING_INFER_REQUESTS, unsigned int);

/**
* @brief Metric to get an unsigned int value of number of infer request in execution stage.
* String value is "NUMBER_OF_EXEC_INFER_REQUESTS". This can be used as an executable network metric as well
*/
DECLARE_METRIC_KEY(NUMBER_OF_EXEC_INFER_REQUESTS, unsigned int);

/**
* @brief Metric to get a name of network. String value is "NETWORK_NAME".
*/
DECLARE_EXEC_NETWORK_METRIC_KEY(NETWORK_NAME, std::string);

/**
 * @brief  Metric to get a float of device thermal. String value is "DEVICE_THERMAL"
 */
DECLARE_METRIC_KEY(DEVICE_THERMAL, float);

/**
* @brief Metric to get an unsigned integer value of optimal number of executable network infer requests.
*/
DECLARE_EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS, unsigned int);

}  // namespace Metrics

/**
 * @brief Generic plugin configuration
 */
namespace PluginConfigParams {

/**
 * @def CONFIG_KEY(name)
 * @brief shortcut for defining configuration keys
 */
#define CONFIG_KEY(name) InferenceEngine::PluginConfigParams::_CONFIG_KEY(name)
#define _CONFIG_KEY(name) KEY_##name
#define DECLARE_CONFIG_KEY(name) static constexpr auto _CONFIG_KEY(name) = #name

/**
 * @def CONFIG_VALUE(name)
 * @brief shortcut for defining configuration values
 */
#define CONFIG_VALUE(name) InferenceEngine::PluginConfigParams::name
#define DECLARE_CONFIG_VALUE(name) static constexpr auto name = #name

/**
* @brief generic boolean values
*/
DECLARE_CONFIG_VALUE(YES);
DECLARE_CONFIG_VALUE(NO);

/**
* @brief Limit #threads that are used by Inference Engine for inference on the CPU.
*/
DECLARE_CONFIG_KEY(CPU_THREADS_NUM);

/**
* @brief The name for setting CPU affinity per thread option.
* It is passed to IInferencePlugin::SetConfig(), this option should be used with values:
* PluginConfigParams::YES or PluginConfigParams::NO
* Ignored, if the OpenVINO compiled with OpenMP threading and any affinity-related OpenMP's
* environment variable is set
*/
DECLARE_CONFIG_KEY(CPU_BIND_THREAD);

/**
* @brief Optimize CPU execution to maximize throughput.
* It is passed to IInferencePlugin::SetConfig(), this option should be used with values:
* - KEY_CPU_THROUGHPUT_NUMA creates as many streams as needed to accomodate NUMA and avoid associated penalties
* - KEY_CPU_THROUGHPUT_AUTO creates bare minimum of streams to improve the performance,
*   this is the most portable option if you have no insights into how many cores you target machine will have
*   (and what is the optimal number of streams)
* - finally, specifying the positive integer value creates the requested number of streams
*/
DECLARE_CONFIG_VALUE(CPU_THROUGHPUT_NUMA);
DECLARE_CONFIG_VALUE(CPU_THROUGHPUT_AUTO);
DECLARE_CONFIG_KEY(CPU_THROUGHPUT_STREAMS);

/**
* @brief Optimize GPU plugin execution to maximize throughput.
* It is passed to IInferencePlugin::SetConfig(), this option should be used with values:
* - KEY_GPU_THROUGHPUT_AUTO creates bare minimum of streams that might improve performance in some cases,
*   this option allows to enable throttle hint for opencl queue thus reduce CPU load without significant performance drop
* - a positive integer value creates the requested number of streams
*/
DECLARE_CONFIG_VALUE(GPU_THROUGHPUT_AUTO);
DECLARE_CONFIG_KEY(GPU_THROUGHPUT_STREAMS);


/**
* @brief The name for setting performance counters option.
* It is passed to IInferencePlugin::SetConfig(), this option should be used with values:
* PluginConfigParams::YES or PluginConfigParams::NO
*/
DECLARE_CONFIG_KEY(PERF_COUNT);

/**
* @brief The key defines dynamic limit of batch processing.
* Specified value is applied to all following Infer() calls. Inference Engine processes
* min(batch_limit, original_batch_size) first pictures from input blob. For example, if input
* blob has sizes 32x3x224x224 after applying plugin.SetConfig({KEY_DYN_BATCH_LIMIT, 10})
* Inference Engine primitives processes only beginner subblobs with size 10x3x224x224.
* This value can be changed before any Infer() call to specify a new batch limit.
*
* The paired parameter value should be convertible to integer number. Acceptable values:
* -1 - Do not limit batch processing
* >0 - Direct value of limit. Batch size to process is min(new batch_limit, original_batch)
*/
DECLARE_CONFIG_KEY(DYN_BATCH_LIMIT);

DECLARE_CONFIG_KEY(DYN_BATCH_ENABLED);

/**
* @brief The key controls threading inside Inference Engine.
* It is passed to IInferencePlugin::SetConfig(), this option should be used with values:
* PluginConfigParams::YES or PluginConfigParams::NO
*/
DECLARE_CONFIG_KEY(SINGLE_THREAD);

/**
* @brief This key directs the plugin to load a configuration file.
* The value should be a file name with the plugin specific configuration
*/
DECLARE_CONFIG_KEY(CONFIG_FILE);

/**
* @brief This key enables dumping of the kernels used by the plugin for custom layers.
* This option should be used with values: PluginConfigParams::YES or PluginConfigParams::NO (default)
*/
DECLARE_CONFIG_KEY(DUMP_KERNELS);

/**
* @brief This key controls performance tuning done or used by the plugin.
* This option should be used with values: PluginConfigParams::TUNING_CREATE,
* PluginConfigParams::TUNING_USE_EXISTING or PluginConfigParams::TUNING_DISABLED (default)
*/
DECLARE_CONFIG_KEY(TUNING_MODE);


DECLARE_CONFIG_VALUE(TUNING_CREATE);
DECLARE_CONFIG_VALUE(TUNING_USE_EXISTING);
DECLARE_CONFIG_VALUE(TUNING_DISABLED);

/**
* @brief This key defines the tuning data filename to be created/used
*/
DECLARE_CONFIG_KEY(TUNING_FILE);

/**
* @brief the key for setting desirable log level.
* This option should be used with values: PluginConfigParams::LOG_NONE (default),
* PluginConfigParams::LOG_WARNING, PluginConfigParams::LOG_INFO, PluginConfigParams::LOG_DEBUG
*/
DECLARE_CONFIG_KEY(LOG_LEVEL);

DECLARE_CONFIG_VALUE(LOG_NONE);
DECLARE_CONFIG_VALUE(LOG_WARNING);
DECLARE_CONFIG_VALUE(LOG_INFO);
DECLARE_CONFIG_VALUE(LOG_DEBUG);

/**
* @brief the key for setting of required device to execute on
* values: device id starts from "0" - first device, "1" - second device, etc
*/
DECLARE_CONFIG_KEY(DEVICE_ID);

/**
* @brief the key for enabling exclusive mode for async requests of different executable networks and the same plugin.
* Sometimes it's necessary to avoid oversubscription requests that are sharing the same device in parallel.
* E.g. There 2 task executors for CPU device: one - in the Hetero plugin, another - in pure CPU plugin.
* Parallel execution both of them might lead to oversubscription and not optimal CPU usage. More efficient
* to run the corresponding tasks one by one via single executor.
* By default, the option is set to YES for hetero cases, and to NO for conventional (single-plugin) cases
* Notice that setting YES disables the CPU streams feature (see another config key in this file)
*/
DECLARE_CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS);

/**
 * @brief This key enables dumping of the internal primitive graph.
 * Should be passed into LoadNetwork method to enable dumping of internal graph of primitives and
 * corresponding configuration information. Value is a name of output dot file without extension.
 * Files <dot_file_name>_init.dot and <dot_file_name>_perf.dot will be produced.
 */
DECLARE_CONFIG_KEY(DUMP_EXEC_GRAPH_AS_DOT);

}  // namespace PluginConfigParams
}  // namespace InferenceEngine
