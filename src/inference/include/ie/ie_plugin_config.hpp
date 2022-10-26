// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware related properties for Inference Engine plugins
 *        To use in SetConfig, LoadNetwork, ImportNetwork methods of plugins
 *
 * @file ie_plugin_config.hpp
 */
#pragma once

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "ie_precision.hpp"

namespace InferenceEngine {

/**
 * @brief %Metrics
 */
namespace Metrics {

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

#ifndef DECLARE_METRIC_KEY_IMPL
#    define DECLARE_METRIC_KEY(name, ...) static constexpr auto METRIC_##name = #    name
#else
#    define DECLARE_METRIC_KEY(name, ...)            \
        static constexpr auto METRIC_##name = #name; \
        DECLARE_METRIC_KEY_IMPL(name, __VA_ARGS__)
#endif

#define DECLARE_EXEC_NETWORK_METRIC_KEY(name, ...) DECLARE_METRIC_KEY(name, __VA_ARGS__)

/**
 * @def METRIC_VALUE(name)
 * @brief shortcut for defining metric values
 */
#define METRIC_VALUE(name)         InferenceEngine::Metrics::name
#define DECLARE_METRIC_VALUE(name) static constexpr auto name = #name

/**
 * @brief Metric to get a std::vector<std::string> of available device IDs. String value is "AVAILABLE_DEVICES"
 */
DECLARE_METRIC_KEY(AVAILABLE_DEVICES, std::vector<std::string>);

/**
 * @brief Metric to get a std::vector<std::string> of supported metrics. String value is "SUPPORTED_METRICS"
 *
 * This can be used as an executable network metric as well.
 *
 * Each of the returned device metrics can be passed to Core::GetMetric, executable network metrics
 * can be passed to ExecutableNetwork::GetMetric.
 *
 */
DECLARE_METRIC_KEY(SUPPORTED_METRICS, std::vector<std::string>);

/**
 * @brief Metric to get a std::vector<std::string> of supported config keys. String value is "SUPPORTED_CONFIG_KEYS"
 *
 * This can be used as an executable network metric as well.
 *
 * Each of the returned device configuration keys can be passed to Core::SetConfig, Core::GetConfig, and
 * Core::LoadNetwork, configuration keys for executable networks can be passed to ExecutableNetwork::SetConfig and
 * ExecutableNetwork::GetConfig.
 *
 */
DECLARE_METRIC_KEY(SUPPORTED_CONFIG_KEYS, std::vector<std::string>);

/**
 * @brief Metric to get a std::string value representing a full device name. String value is "FULL_DEVICE_NAME"
 */
DECLARE_METRIC_KEY(FULL_DEVICE_NAME, std::string);

/**
 * @brief Metric to get a std::vector<std::string> of optimization options per device. String value is
 * "OPTIMIZATION_CAPABILITIES"
 *
 * The possible values:
 *  - "FP32" - device can support FP32 models
 *  - "BF16" - device can support BF16 computations for models
 *  - "FP16" - device can support FP16 models
 *  - "INT8" - device can support models with INT8 layers
 *  - "BIN" - device can support models with BIN layers
 *  - "WINOGRAD" - device can support models where convolution implemented via Winograd transformations
 *  - "BATCHED_BLOB" - device can support BatchedBlob
 */
DECLARE_METRIC_KEY(OPTIMIZATION_CAPABILITIES, std::vector<std::string>);

DECLARE_METRIC_VALUE(FP32);
DECLARE_METRIC_VALUE(BF16);
DECLARE_METRIC_VALUE(FP16);
DECLARE_METRIC_VALUE(INT8);
DECLARE_METRIC_VALUE(BIN);
DECLARE_METRIC_VALUE(WINOGRAD);
DECLARE_METRIC_VALUE(BATCHED_BLOB);

/**
 * @brief Metric to provide information about a range for streams on platforms where streams are supported.
 *
 * Metric returns a value of std::tuple<unsigned int, unsigned int> type, where:
 *  - First value is bottom bound.
 *  - Second value is upper bound.
 * String value for metric name is "RANGE_FOR_STREAMS".
 */
DECLARE_METRIC_KEY(RANGE_FOR_STREAMS, std::tuple<unsigned int, unsigned int>);
/**
 * @brief Metric to query information optimal batch size for the given device and the network
 *
 * Metric returns a value of unsigned int type,
 * Returns optimal batch size for a given network on the given device. The returned value is aligned to power of 2.
 * Also, MODEL_PTR is the required option for this metric since the optimal batch size depends on the model,
 * so if the MODEL_PTR is not given, the result of the metric is always 1.
 * For the GPU the metric is queried automatically whenever the OpenVINO performance hint for the throughput is used,
 * so that the result (>1) governs the automatic batching (transparently to the application).
 * The automatic batching can be disabled with ALLOW_AUTO_BATCHING set to NO
 */
DECLARE_METRIC_KEY(OPTIMAL_BATCH_SIZE, unsigned int);

/**
 * @brief Metric to get maximum batch size which does not cause performance degradation due to memory swap impact.
 *
 * Metric returns a value of unsigned int type,
 * Note that the returned value may not aligned to power of 2.
 * Also, MODEL_PTR is the required option for this metric since the available max batch size depends on the model size.
 * If the MODEL_PTR is not given, it will return 1.
 */
DECLARE_METRIC_KEY(MAX_BATCH_SIZE, unsigned int);

/**
 * @brief Metric to provide a hint for a range for number of async infer requests. If device supports streams,
 * the metric provides range for number of IRs per stream.
 *
 * Metric returns a value of std::tuple<unsigned int, unsigned int, unsigned int> type, where:
 *  - First value is bottom bound.
 *  - Second value is upper bound.
 *  - Third value is step inside this range.
 * String value for metric name is "RANGE_FOR_ASYNC_INFER_REQUESTS".
 */
DECLARE_METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS, std::tuple<unsigned int, unsigned int, unsigned int>);

/**
 * @brief Metric to get an unsigned int value of number of waiting infer request.
 *
 * String value is "NUMBER_OF_WAITNING_INFER_REQUESTS". This can be used as an executable network metric as well
 */
DECLARE_METRIC_KEY(NUMBER_OF_WAITING_INFER_REQUESTS, unsigned int);

/**
 * @brief Metric to get an unsigned int value of number of infer request in execution stage.
 *
 * String value is "NUMBER_OF_EXEC_INFER_REQUESTS". This can be used as an executable network metric as well
 */
DECLARE_METRIC_KEY(NUMBER_OF_EXEC_INFER_REQUESTS, unsigned int);

/**
 * @brief Metric which defines the device architecture.
 */
DECLARE_METRIC_KEY(DEVICE_ARCHITECTURE, std::string);

/**
 * @brief Enum to define possible device types
 */
enum class DeviceType {
    integrated = 0,
    discrete = 1,
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const InferenceEngine::Metrics::DeviceType& deviceType) {
    switch (deviceType) {
    case InferenceEngine::Metrics::DeviceType::discrete:
        os << "discrete";
        break;
    case InferenceEngine::Metrics::DeviceType::integrated:
        os << "integrated";
        break;
    default:
        os << "unknown";
        break;
    }

    return os;
}
/** @endcond */

/**
 * @brief Metric to get a type of device. See DeviceType enum definition for possible return values
 */
DECLARE_METRIC_KEY(DEVICE_TYPE, DeviceType);

/**
 * @brief Metric which defines Giga OPS per second count (GFLOPS or GIOPS) for a set of precisions supported by
 * specified device
 */
DECLARE_METRIC_KEY(DEVICE_GOPS, std::map<InferenceEngine::Precision, float>);

/**
 * @brief Metric which defines support of import/export functionality by plugin
 */
DECLARE_METRIC_KEY(IMPORT_EXPORT_SUPPORT, bool);

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
#define CONFIG_KEY(name)         InferenceEngine::PluginConfigParams::_CONFIG_KEY(name)
#define _CONFIG_KEY(name)        KEY_##name
#define DECLARE_CONFIG_KEY(name) static constexpr auto _CONFIG_KEY(name) = #name

/**
 * @def CONFIG_VALUE(name)
 * @brief shortcut for defining configuration values
 */
#define CONFIG_VALUE(name)         InferenceEngine::PluginConfigParams::name
#define DECLARE_CONFIG_VALUE(name) static constexpr auto name = #name

/**
 * @brief (Optional) config key that defines what model should be provided with more performant bounded resource first
 * It provides 3 types of levels: High, Medium and Low. The default value is Medium
 */
DECLARE_CONFIG_KEY(MODEL_PRIORITY);
DECLARE_CONFIG_VALUE(MODEL_PRIORITY_HIGH);
DECLARE_CONFIG_VALUE(MODEL_PRIORITY_MED);
DECLARE_CONFIG_VALUE(MODEL_PRIORITY_LOW);

/**
 * @brief High-level OpenVINO Performance Hints
 * unlike low-level config keys that are individual (per-device), the hints are smth that every device accepts
 * and turns into device-specific settings
 */
DECLARE_CONFIG_KEY(PERFORMANCE_HINT);
DECLARE_CONFIG_VALUE(LATENCY);
DECLARE_CONFIG_VALUE(THROUGHPUT);
DECLARE_CONFIG_VALUE(CUMULATIVE_THROUGHPUT);
/**
 * @brief (Optional) config key that backs the (above) Performance Hints
 * by giving additional information on how many inference requests the application will be keeping in flight
 * usually this value comes from the actual use-case (e.g. number of video-cameras, or other sources of inputs)
 */
DECLARE_CONFIG_KEY(PERFORMANCE_HINT_NUM_REQUESTS);
/**
 * @brief (Optional) config key that governs Auto-Batching (with YES/NO values, below)
 */
DECLARE_CONFIG_KEY(ALLOW_AUTO_BATCHING);

/**
 * @brief generic boolean values
 */
DECLARE_CONFIG_VALUE(YES);
DECLARE_CONFIG_VALUE(NO);

/**
 * @brief Auto-batching configuration, string for the device + batch size, e.g. "GPU(4)"
 */
DECLARE_CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG);
/**
 * @brief Auto-batching configuration: string with timeout (in ms), e.g. "100"
 */
DECLARE_CONFIG_KEY(AUTO_BATCH_TIMEOUT);

/**
 * @brief Limit `#threads` that are used by Inference Engine for inference on the CPU.
 */
DECLARE_CONFIG_KEY(CPU_THREADS_NUM);

/**
 * @brief The name for setting CPU affinity per thread option.
 *
 * It is passed to Core::SetConfig(), this option should be used with values:
 * PluginConfigParams::NO (no pinning for CPU inference threads)
 * PluginConfigParams::YES, which is default on the conventional CPUs (pinning threads to cores, best for static
 * benchmarks),
 *
 * the following options are implemented only for the TBB as a threading option
 * PluginConfigParams::NUMA (pinning threads to NUMA nodes, best for real-life, contented cases)
 *      on the Windows and MacOS* this option behaves as YES
 * PluginConfigParams::HYBRID_AWARE (let the runtime to do pinning to the cores types, e.g. prefer the "big" cores for
 * latency tasks) on the hybrid CPUs this option is default
 *
 * Also, the settings are ignored, if the OpenVINO compiled with OpenMP and any affinity-related OpenMP's
 * environment variable is set (as affinity is configured explicitly)
 */
DECLARE_CONFIG_KEY(CPU_BIND_THREAD);
DECLARE_CONFIG_VALUE(NUMA);
DECLARE_CONFIG_VALUE(HYBRID_AWARE);

/**
 * @brief Optimize CPU execution to maximize throughput.
 *
 * It is passed to Core::SetConfig(), this option should be used with values:
 * - KEY_CPU_THROUGHPUT_NUMA creates as many streams as needed to accommodate NUMA and avoid associated penalties
 * - KEY_CPU_THROUGHPUT_AUTO creates bare minimum of streams to improve the performance,
 *   this is the most portable option if you have no insights into how many cores you target machine will have
 *   (and what is the optimal number of streams)
 * - finally, specifying the positive integer value creates the requested number of streams
 */
DECLARE_CONFIG_KEY(CPU_THROUGHPUT_STREAMS);
DECLARE_CONFIG_VALUE(CPU_THROUGHPUT_NUMA);
DECLARE_CONFIG_VALUE(CPU_THROUGHPUT_AUTO);

/**
 * @brief The name for setting performance counters option.
 *
 * It is passed to Core::SetConfig(), this option should be used with values:
 * PluginConfigParams::YES or PluginConfigParams::NO
 */
DECLARE_CONFIG_KEY(PERF_COUNT);

/**
 * @brief The key defines dynamic limit of batch processing.
 *
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

/**
 * @brief The key checks whether dynamic batch is enabled.
 */
DECLARE_CONFIG_KEY(DYN_BATCH_ENABLED);

/**
 * @brief This key directs the plugin to load a configuration file.
 *
 * The value should be a file name with the plugin specific configuration
 */
DECLARE_CONFIG_KEY(CONFIG_FILE);

/**
 * @brief the key for setting desirable log level.
 *
 * This option should be used with values: PluginConfigParams::LOG_NONE (default),
 * PluginConfigParams::LOG_ERROR, PluginConfigParams::LOG_WARNING,
 * PluginConfigParams::LOG_INFO, PluginConfigParams::LOG_DEBUG, PluginConfigParams::LOG_TRACE
 */
DECLARE_CONFIG_KEY(LOG_LEVEL);

DECLARE_CONFIG_VALUE(LOG_NONE);     // turn off logging
DECLARE_CONFIG_VALUE(LOG_ERROR);    // error events that might still allow the
                                    // application to continue running
DECLARE_CONFIG_VALUE(LOG_WARNING);  // potentially harmful situations which may
                                    // further lead to ERROR
DECLARE_CONFIG_VALUE(LOG_INFO);     // informational messages that display the progress of the
                                    // application at coarse-grained level
DECLARE_CONFIG_VALUE(LOG_DEBUG);    // fine-grained events that are most useful to
                                    // debug an application.
DECLARE_CONFIG_VALUE(LOG_TRACE);    // finer-grained informational events than the DEBUG

/**
 * @brief the key for setting of required device to execute on
 * values: device id starts from "0" - first device, "1" - second device, etc
 */
DECLARE_CONFIG_KEY(DEVICE_ID);

/**
 * @brief the key for enabling exclusive mode for async requests of different executable networks and the same plugin.
 *
 * Sometimes it is necessary to avoid oversubscription requests that are sharing the same device in parallel.
 * E.g. There 2 task executors for CPU device: one - in the Hetero plugin, another - in pure CPU plugin.
 * Parallel execution both of them might lead to oversubscription and not optimal CPU usage. More efficient
 * to run the corresponding tasks one by one via single executor.
 * By default, the option is set to YES for hetero cases, and to NO for conventional (single-plugin) cases
 * Notice that setting YES disables the CPU streams feature (see another config key in this file)
 */
DECLARE_CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS);

/**
 * @deprecated Use InferenceEngine::ExecutableNetwork::GetExecGraphInfo::serialize method
 * @brief This key enables dumping of the internal primitive graph.
 *
 * Should be passed into LoadNetwork method to enable dumping of internal graph of primitives and
 * corresponding configuration information. Value is a name of output dot file without extension.
 * Files `<dot_file_name>_init.dot` and `<dot_file_name>_perf.dot` will be produced.
 */
INFERENCE_ENGINE_DEPRECATED("Use InferenceEngine::ExecutableNetwork::GetExecGraphInfo::serialize method")
DECLARE_CONFIG_KEY(DUMP_EXEC_GRAPH_AS_DOT);

/**
 * @brief The name for setting to execute in bfloat16 precision whenever it is possible
 *
 * This option let plugin know to downscale the precision where it see performance benefits from
 * bfloat16 execution
 * Such option do not guarantee accuracy of the network, the accuracy in this mode should be
 * verified separately by the user and basing on performance and accuracy results it should be
 * user's decision to use this option or not to use
 */
DECLARE_CONFIG_KEY(ENFORCE_BF16);

/**
 * @brief This key defines the directory which will be used to store any data cached by plugins.
 *
 * The underlying cache structure is not defined and might differ between OpenVINO releases
 * Cached data might be platform / device specific and might be invalid after OpenVINO version change
 * If this key is not specified or value is empty string, then caching is disabled.
 * The key might enable caching for the plugin using the following code:
 *
 * @code
 * ie.SetConfig({{CONFIG_KEY(CACHE_DIR), "cache/"}}, "GPU"); // enables cache for GPU plugin
 * @endcode
 *
 * The following code enables caching of compiled network blobs for devices where import/export is supported
 *
 * @code
 * ie.SetConfig({{CONFIG_KEY(CACHE_DIR), "cache/"}}); // enables models cache
 * @endcode
 */
DECLARE_CONFIG_KEY(CACHE_DIR);

/**
 * @brief The key to decide whether terminate tbb threads when inference engine destructing.
 *
 * value type: boolean
 *   - True explicitly terminate tbb when inference engine destruction
 *   - False will not involve additional tbb operations when inference engine destruction
 *
 * @code
 * ie.SetConfig({{CONFIG_KEY(FORCE_TBB_TERMINATE), CONFIG_VALUE(YES)}}); // enable
 * @endcode
 */
DECLARE_CONFIG_KEY(FORCE_TBB_TERMINATE);

}  // namespace PluginConfigParams

/**
 * @def AUTO_CONFIG_KEY(name)
 * @brief A macro which provides an AUTO-mangled name for configuration key with name `name`
 */
#define AUTO_CONFIG_KEY(name) InferenceEngine::_CONFIG_KEY(AUTO_##name)

#define DECLARE_AUTO_CONFIG_KEY(name) DECLARE_CONFIG_KEY(AUTO_##name)

}  // namespace InferenceEngine

#include "hetero/hetero_plugin_config.hpp"
#include "multi-device/multi_device_config.hpp"

// remove in 2022.1 major release
#include "cldnn/cldnn_config.hpp"
#include "gna/gna_config.hpp"
