// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sys/stat.h>

#include <cldnn/cldnn_config.hpp>
#include <gpu/gpu_config.hpp>
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "ie_api.h"
#include "file_utils.h"
#include "intel_gpu/plugin/device_config.hpp"
#include "intel_gpu/plugin/itt.hpp"
#include <ie_system_conf.h>
#include <thread>

#ifdef _WIN32
# include <direct.h>
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
# define mkdir(dir, mode) _wmkdir(dir)
#else
# define mkdir(dir, mode) _mkdir(dir)
#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#endif  // _WIN32

using namespace InferenceEngine;

namespace ov {
namespace runtime {
namespace intel_gpu {

static void createDirectory(std::string _path) {
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring widepath = ov::util::string_to_wstring(_path.c_str());
    const wchar_t* path = widepath.c_str();
#else
    const char* path = _path.c_str();
#endif

    auto err = mkdir(path, 0755);
    if (err != 0 && errno != EEXIST) {
        IE_THROW() << "Couldn't create directory! (err=" << err << "; errno=" << errno << ")";
    }
}

static int getNumberOfCores(const IStreamsExecutor::Config::PreferredCoreType core_type) {
    const auto total_num_cores = getNumberOfLogicalCPUCores();
    const auto total_num_big_cores = getNumberOfLogicalCPUCores(true);
    const auto total_num_little_cores = total_num_cores - total_num_big_cores;

    int num_cores = total_num_cores;
    if (core_type == IStreamsExecutor::Config::BIG) {
        num_cores = total_num_big_cores;
    } else if (core_type == IStreamsExecutor::Config::LITTLE) {
        num_cores = total_num_little_cores;
    }
    return num_cores;
}

IE_SUPPRESS_DEPRECATED_START
void Config::UpdateFromMap(const std::map<std::string, std::string>& configMap) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Config::UpdateFromMap");
    for (auto& kvp : configMap) {
        std::string key = kvp.first;
        std::string val = kvp.second;
        const auto hints = perfHintsConfig.SupportedKeys();
        if (hints.end() != std::find(hints.begin(), hints.end(), key)) {
            perfHintsConfig.SetConfig(key, val);
        } else if (key.compare(PluginConfigParams::KEY_PERF_COUNT) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                useProfiling = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                useProfiling = false;
            } else {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_DYN_BATCH_ENABLED) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                enableDynamicBatch = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                enableDynamicBatch = false;
            } else {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_DUMP_KERNELS) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                dumpCustomKernels = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                dumpCustomKernels = false;
            } else {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(GPUConfigParams::KEY_GPU_PLUGIN_PRIORITY) == 0 ||
                   key.compare(CLDNNConfigParams::KEY_CLDNN_PLUGIN_PRIORITY) == 0) {
            std::stringstream ss(val);
            uint32_t uVal(0);
            ss >> uVal;
            if (ss.fail()) {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
            switch (uVal) {
                case 0:
                    queuePriority = cldnn::priority_mode_types::disabled;
                    break;
                case 1:
                    queuePriority = cldnn::priority_mode_types::low;
                    break;
                case 2:
                    queuePriority = cldnn::priority_mode_types::med;
                    break;
                case 3:
                    queuePriority = cldnn::priority_mode_types::high;
                    break;
                default:
                    IE_THROW(ParameterMismatch) << "Unsupported queue priority value: " << uVal;
            }
        } else if (key.compare(GPUConfigParams::KEY_GPU_MODEL_PRIORITY) == 0) {
            bool found_matched_value = false;
            if (val.find(GPUConfigParams::GPU_MODEL_PRIORITY_HIGH) != std::string::npos) {
                queuePriority = cldnn::priority_mode_types::high;
                task_exec_config._threadPreferredCoreType = IStreamsExecutor::Config::BIG;
                found_matched_value = true;
            } else if (val.find(GPUConfigParams::GPU_MODEL_PRIORITY_LOW) != std::string::npos) {
                queuePriority = cldnn::priority_mode_types::low;
                task_exec_config._threadPreferredCoreType = IStreamsExecutor::Config::LITTLE;
                found_matched_value = true;
            } else {
                if (val.find(GPUConfigParams::GPU_QUEUE_PRIORITY_HIGH) != std::string::npos) {
                    queuePriority = cldnn::priority_mode_types::high;
                    found_matched_value = true;
                } else if (val.find(GPUConfigParams::GPU_QUEUE_PRIORITY_MED) != std::string::npos) {
                    queuePriority = cldnn::priority_mode_types::med;
                    found_matched_value = true;
                } else if (val.find(GPUConfigParams::GPU_QUEUE_PRIORITY_LOW) != std::string::npos) {
                    queuePriority = cldnn::priority_mode_types::low;
                    found_matched_value = true;
                } else if (val.find(GPUConfigParams::GPU_QUEUE_PRIORITY_DEFAULT) != std::string::npos) {
                    queuePriority = cldnn::priority_mode_types::disabled;
                    found_matched_value = true;
                } else { // default is disabled
                    queuePriority = cldnn::priority_mode_types::disabled;
                }
                if (val.find(GPUConfigParams::GPU_HOST_TASK_PRIORITY_HIGH) != std::string::npos) {
                    task_exec_config._threadPreferredCoreType = IStreamsExecutor::Config::BIG;
                    found_matched_value = true;
                } else if (val.find(GPUConfigParams::GPU_HOST_TASK_PRIORITY_LOW) != std::string::npos) {
                    task_exec_config._threadPreferredCoreType = IStreamsExecutor::Config::LITTLE;
                    found_matched_value = true;
                } else if (val.find(GPUConfigParams::GPU_HOST_TASK_PRIORITY_ANY) != std::string::npos) {
                    task_exec_config._threadPreferredCoreType = IStreamsExecutor::Config::ANY;
                    found_matched_value = true;
                } else { // default is any
                    task_exec_config._threadPreferredCoreType = IStreamsExecutor::Config::ANY;
                }
            }
            if (!found_matched_value) {
                IE_THROW() << "Not found appropriate value for property key " << GPUConfigParams::KEY_GPU_PLUGIN_PRIORITY
                    << ".\n Expected Plugin priority such as GPU_PLUGIN_PRIORITY_HIGH / GPU_PLUGIN_PRIORITY_LOW or\n"
                    << " Combination of queue priority(HIGH, MED, LOW, and DISABLED) and host task priority(HIGH, LOW, and ANY)"
                    << " such as GPU_QUEUE_PRIORITY_HIGH | GPU_HOST_TASK_PRIORITY_HIGH";
            }

            if (getAvailableCoresTypes().size() > 1) {
                if (task_exec_config._threadPreferredCoreType == IStreamsExecutor::Config::BIG
                    || task_exec_config._threadPreferredCoreType == IStreamsExecutor::Config::LITTLE) {
                        task_exec_config._streams = std::min(task_exec_config._streams,
                                                        getNumberOfCores(task_exec_config._threadPreferredCoreType));
                    }
            } else {
                task_exec_config._threadPreferredCoreType = IStreamsExecutor::Config::ANY;
                task_exec_config._streams = std::min(task_exec_config._streams,
                                                        static_cast<int>(std::thread::hardware_concurrency()));
            }
        } else if (key.compare(GPUConfigParams::KEY_GPU_PLUGIN_THROTTLE) == 0 ||
                   key.compare(CLDNNConfigParams::KEY_CLDNN_PLUGIN_THROTTLE) == 0) {
            std::stringstream ss(val);
            uint32_t uVal(0);
            ss >> uVal;
            if (ss.fail()) {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
            switch (uVal) {
                case 0:
                    queueThrottle = cldnn::throttle_mode_types::disabled;
                    break;
                case 1:
                    queueThrottle = cldnn::throttle_mode_types::low;
                    break;
                case 2:
                    queueThrottle = cldnn::throttle_mode_types::med;
                    break;
                case 3:
                    queueThrottle = cldnn::throttle_mode_types::high;
                    break;
                default:
                    IE_THROW(ParameterMismatch) << "Unsupported queue throttle value: " << uVal;
            }
        } else if (key.compare(PluginConfigParams::KEY_CONFIG_FILE) == 0) {
            std::stringstream ss(val);
            std::istream_iterator<std::string> begin(ss);
            std::istream_iterator<std::string> end;
            std::vector<std::string> configFiles(begin, end);
            for (auto& file : configFiles) {
                CustomLayer::LoadFromFile(file, customLayers);
            }
        } else if (key.compare(PluginConfigParams::KEY_TUNING_MODE) == 0) {
            if (val.compare(PluginConfigParams::TUNING_DISABLED) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_disabled;
            } else if (val.compare(PluginConfigParams::TUNING_CREATE) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_tune_and_cache;
            } else if (val.compare(PluginConfigParams::TUNING_USE_EXISTING) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_use_cache;
            } else if (val.compare(PluginConfigParams::TUNING_UPDATE) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_use_and_update;
            } else if (val.compare(PluginConfigParams::TUNING_RETUNE) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_retune_and_cache;
            } else {
                IE_THROW(NotFound) << "Unsupported tuning mode value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_TUNING_FILE) == 0) {
            tuningConfig.cache_file_path = val;
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_MEM_POOL) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                memory_pool_on = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                memory_pool_on = false;
            } else {
                IE_THROW(NotFound) << "Unsupported memory pool flag value: " << val;
            }
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_GRAPH_DUMPS_DIR) == 0) {
            if (!val.empty()) {
                graph_dumps_dir = val;
                createDirectory(graph_dumps_dir);
            }
        } else if (key.compare(PluginConfigParams::KEY_CACHE_DIR) == 0) {
            if (!val.empty()) {
                kernels_cache_dir = val;
                createDirectory(kernels_cache_dir);
            }
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_SOURCES_DUMPS_DIR) == 0) {
            if (!val.empty()) {
                sources_dumps_dir = val;
                createDirectory(sources_dumps_dir);
            }
        } else if (key.compare(PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                exclusiveAsyncRequests = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                exclusiveAsyncRequests = false;
            } else {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS) == 0) {
            if (val.compare(PluginConfigParams::GPU_THROUGHPUT_AUTO) == 0) {
                throughput_streams = GetDefaultNStreamsForThroughputMode();
            } else {
                int val_i;
                try {
                    val_i = std::stoi(val);
                } catch (const std::exception&) {
                    IE_THROW() << "Wrong value for property key " << PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS
                                       << ". Expected only positive numbers (#streams) or "
                                       << "PluginConfigParams::GPU_THROUGHPUT_AUTO";
                }
                if (val_i > 0)
                    throughput_streams = static_cast<uint16_t>(val_i);
            }
        } else if (key.compare(PluginConfigParams::KEY_DEVICE_ID) == 0) {
            // Validate if passed value is postivie number.
            try {
                int val_i = std::stoi(val);
                (void)val_i;
            } catch (const std::exception&) {
                IE_THROW() << "Wrong value for property key " << PluginConfigParams::KEY_DEVICE_ID
                    << ". DeviceIDs are only represented by positive numbers";
            }
            // Set this value.
            device_id = val;
        } else if (key.compare(PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                enableInt8 = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                enableInt8 = false;
            } else {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(GPUConfigParams::KEY_GPU_NV12_TWO_INPUTS) == 0 ||
                   key.compare(CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                nv12_two_inputs = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                nv12_two_inputs = false;
            } else {
                IE_THROW(NotFound) << "Unsupported NV12 flag value: " << val;
            }
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_ENABLE_FP16_FOR_QUANTIZED_MODELS) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                enable_fp16_for_quantized_models = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                enable_fp16_for_quantized_models = false;
            } else {
                IE_THROW(NotFound) << "Unsupported KEY_CLDNN_ENABLE_FP16_FOR_QUANTIZED_MODELS flag value: " << val;
            }
        } else if (key.compare(GPUConfigParams::KEY_GPU_MAX_NUM_THREADS) == 0) {
            int max_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
            try {
                int val_i = std::stoi(val);
                if (val_i <= 0 || val_i > max_threads) {
                    val_i = max_threads;
                }
                task_exec_config._streams = std::min(task_exec_config._streams, val_i);
            } catch (const std::exception&) {
                IE_THROW() << "Wrong value for property key " << GPUConfigParams::KEY_GPU_MAX_NUM_THREADS << ": " << val
                                   << "\nSpecify the number of threads use for build as an integer."
                                   << "\nOut of range value will be set as a default value, maximum concurrent threads.";
            }
        } else if (key.compare(GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                enable_loop_unrolling = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                enable_loop_unrolling = false;
            } else {
                IE_THROW(ParameterMismatch) << "Unsupported KEY_GPU_ENABLE_LOOP_UNROLLING flag value: " << val;
            }
        } else {
            IE_THROW(NotFound) << "Unsupported property key by plugin: " << key;
        }

        adjustKeyMapValues();
    }
}

void Config::adjustKeyMapValues() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Config::AdjustKeyMapValues");
    if (useProfiling)
        key_config_map[PluginConfigParams::KEY_PERF_COUNT] = PluginConfigParams::YES;
    else
        key_config_map[PluginConfigParams::KEY_PERF_COUNT] = PluginConfigParams::NO;

    if (dumpCustomKernels)
        key_config_map[PluginConfigParams::KEY_DUMP_KERNELS] = PluginConfigParams::YES;
    else
        key_config_map[PluginConfigParams::KEY_DUMP_KERNELS] = PluginConfigParams::NO;

    if (exclusiveAsyncRequests)
        key_config_map[PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS] = PluginConfigParams::YES;
    else
        key_config_map[PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS] = PluginConfigParams::NO;

    if (memory_pool_on)
        key_config_map[CLDNNConfigParams::KEY_CLDNN_MEM_POOL] = PluginConfigParams::YES;
    else
        key_config_map[CLDNNConfigParams::KEY_CLDNN_MEM_POOL] = PluginConfigParams::NO;

    if (enableDynamicBatch)
        key_config_map[PluginConfigParams::KEY_DYN_BATCH_ENABLED] = PluginConfigParams::YES;
    else
        key_config_map[PluginConfigParams::KEY_DYN_BATCH_ENABLED] = PluginConfigParams::NO;

    if (nv12_two_inputs) {
        key_config_map[CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS] = PluginConfigParams::YES;
        key_config_map[GPUConfigParams::KEY_GPU_NV12_TWO_INPUTS] = PluginConfigParams::YES;
    } else {
        key_config_map[CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS] = PluginConfigParams::NO;
        key_config_map[GPUConfigParams::KEY_GPU_NV12_TWO_INPUTS] = PluginConfigParams::NO;
    }

    if (enable_fp16_for_quantized_models)
        key_config_map[CLDNNConfigParams::KEY_CLDNN_ENABLE_FP16_FOR_QUANTIZED_MODELS] = PluginConfigParams::YES;
    else
        key_config_map[CLDNNConfigParams::KEY_CLDNN_ENABLE_FP16_FOR_QUANTIZED_MODELS] = PluginConfigParams::NO;

    {
        if (queuePriority == cldnn::priority_mode_types::high && task_exec_config._threadPreferredCoreType == IStreamsExecutor::Config::BIG) {
            key_config_map[GPUConfigParams::KEY_GPU_MODEL_PRIORITY] = GPUConfigParams::GPU_MODEL_PRIORITY_HIGH;
        } else if (queuePriority == cldnn::priority_mode_types::low && task_exec_config._threadPreferredCoreType == IStreamsExecutor::Config::LITTLE) {
            key_config_map[GPUConfigParams::KEY_GPU_MODEL_PRIORITY] = GPUConfigParams::GPU_MODEL_PRIORITY_LOW;
        } else {
            std::string val_plugin_priority;
            switch (queuePriority) {
            case cldnn::priority_mode_types::low:   val_plugin_priority = GPUConfigParams::GPU_QUEUE_PRIORITY_LOW; break;
            case cldnn::priority_mode_types::med:   val_plugin_priority = GPUConfigParams::GPU_QUEUE_PRIORITY_MED; break;
            case cldnn::priority_mode_types::high:  val_plugin_priority = GPUConfigParams::GPU_QUEUE_PRIORITY_HIGH; break;
            default:                                val_plugin_priority = GPUConfigParams::GPU_QUEUE_PRIORITY_DEFAULT; break;
            }
            val_plugin_priority += "|";
            switch (task_exec_config._threadPreferredCoreType) {
            case IStreamsExecutor::Config::LITTLE:      val_plugin_priority += GPUConfigParams::GPU_HOST_TASK_PRIORITY_HIGH; break;
            case IStreamsExecutor::Config::BIG:         val_plugin_priority += GPUConfigParams::GPU_HOST_TASK_PRIORITY_LOW; break;
            case IStreamsExecutor::Config::ANY:default: val_plugin_priority += GPUConfigParams::GPU_HOST_TASK_PRIORITY_ANY; break;
            }
            key_config_map[GPUConfigParams::KEY_GPU_PLUGIN_PRIORITY]        = val_plugin_priority;
        }
    }
    {
        std::string qp = "0";
        switch (queuePriority) {
        case cldnn::priority_mode_types::low: qp = "1"; break;
        case cldnn::priority_mode_types::med: qp = "2"; break;
        case cldnn::priority_mode_types::high: qp = "3"; break;
        default: break;
        }
        key_config_map[CLDNNConfigParams::KEY_CLDNN_PLUGIN_PRIORITY] = qp;
        key_config_map[GPUConfigParams::KEY_GPU_PLUGIN_PRIORITY] = qp;
    }
    {
        std::string qt = "0";
        switch (queueThrottle) {
        case cldnn::throttle_mode_types::low: qt = "1"; break;
        case cldnn::throttle_mode_types::med: qt = "2"; break;
        case cldnn::throttle_mode_types::high: qt = "3"; break;
        default: break;
        }
        key_config_map[CLDNNConfigParams::KEY_CLDNN_PLUGIN_THROTTLE] = qt;
        key_config_map[GPUConfigParams::KEY_GPU_PLUGIN_THROTTLE] = qt;
    }
    {
        std::string tm = PluginConfigParams::TUNING_DISABLED;
        switch (tuningConfig.mode) {
        case cldnn::tuning_mode::tuning_tune_and_cache: tm = PluginConfigParams::TUNING_CREATE; break;
        case cldnn::tuning_mode::tuning_use_cache: tm = PluginConfigParams::TUNING_USE_EXISTING; break;
        case cldnn::tuning_mode::tuning_use_and_update: tm = PluginConfigParams::TUNING_UPDATE; break;
        case cldnn::tuning_mode::tuning_retune_and_cache: tm = PluginConfigParams::TUNING_RETUNE; break;
        default: break;
        }
        key_config_map[PluginConfigParams::KEY_TUNING_MODE] = tm;
        key_config_map[PluginConfigParams::KEY_TUNING_FILE] = tuningConfig.cache_file_path;
    }

    key_config_map[CLDNNConfigParams::KEY_CLDNN_GRAPH_DUMPS_DIR] = graph_dumps_dir;
    key_config_map[CLDNNConfigParams::KEY_CLDNN_SOURCES_DUMPS_DIR] = sources_dumps_dir;
    key_config_map[PluginConfigParams::KEY_CACHE_DIR] = kernels_cache_dir;

    key_config_map[PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS] = std::to_string(throughput_streams);
    key_config_map[PluginConfigParams::KEY_DEVICE_ID] = device_id;
    key_config_map[PluginConfigParams::KEY_CONFIG_FILE] = "";
    key_config_map[GPUConfigParams::KEY_GPU_MAX_NUM_THREADS] = std::to_string(task_exec_config._streams);

    if (enable_loop_unrolling)
        key_config_map[GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING] = PluginConfigParams::YES;
    else
        key_config_map[GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING] = PluginConfigParams::NO;
    key_config_map[PluginConfigParams::KEY_PERFORMANCE_HINT]= perfHintsConfig.ovPerfHint;
    key_config_map[PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS] =
        std::to_string(perfHintsConfig.ovPerfHintNumRequests);
}

void Configs::CreateConfig(std::string device_id) {
    if (configs.find(device_id) == configs.end()) {
        configs.emplace(device_id, Config(device_id));
    }
}

Config& Configs::GetConfig(std::string device_id) {
    if (device_id.empty()) {
        return GetDefaultDeviceConfig();
    }
    if (configs.find(device_id) == configs.end()) {
        IE_THROW() << "Config for device with " << device_id << " ID is not registered in GPU plugin";
    }
    return configs.find(device_id)->second;
}

Config& Configs::GetDefaultDeviceConfig() {
    return GetConfig(default_device_id);
}

IE_SUPPRESS_DEPRECATED_END

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
