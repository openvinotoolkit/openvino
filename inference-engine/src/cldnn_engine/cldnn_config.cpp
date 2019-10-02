// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sys/stat.h>

#include <cldnn/cldnn_config.hpp>
#include "cldnn_config.h"
#include "cpp_interfaces/exception2status.hpp"

#if defined(_WIN32)
#define mkdir(dir, mode) _mkdir(dir)
#endif

using namespace InferenceEngine;

namespace CLDNNPlugin {

void Config::UpdateFromMap(const std::map<std::string, std::string>& configMap) {
    for (auto& kvp : configMap) {
        std::string key = kvp.first;
        std::string val = kvp.second;

        if (key.compare(PluginConfigParams::KEY_PERF_COUNT) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                useProfiling = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                useProfiling = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_DYN_BATCH_ENABLED) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                enableDynamicBatch = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                enableDynamicBatch = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_DUMP_KERNELS) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                dumpCustomKernels = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                dumpCustomKernels = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_PLUGIN_PRIORITY) == 0) {
            std::stringstream ss(val);
            uint32_t uVal(0);
            ss >> uVal;
            if (ss.fail()) {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
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
                    THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported queue priority value: " << uVal;
            }

        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_PLUGIN_THROTTLE) == 0) {
            std::stringstream ss(val);
            uint32_t uVal(0);
            ss >> uVal;
            if (ss.fail()) {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
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
                    THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported queue throttle value: " << uVal;
            }
        } else if (key.compare(PluginConfigParams::KEY_CONFIG_FILE) == 0) {
            std::stringstream ss(val);
            std::istream_iterator<std::string> begin(ss);
            std::istream_iterator<std::string> end;
            std::vector<std::string> configFiles(begin, end);
            for (auto& file : configFiles) {
                CLDNNCustomLayer::LoadFromFile(file, customLayers);
            }
        } else if (key.compare(PluginConfigParams::KEY_TUNING_MODE) == 0) {
            if (val.compare(PluginConfigParams::TUNING_DISABLED) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_disabled;
            } else if (val.compare(PluginConfigParams::TUNING_CREATE) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_tune_and_cache;
            } else if (val.compare(PluginConfigParams::TUNING_USE_EXISTING) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_use_cache;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported tuning mode value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_TUNING_FILE) == 0) {
            tuningConfig.cache_file_path = val;
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_MEM_POOL) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                memory_pool_on = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                memory_pool_on = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported memory pool flag value: " << val;
            }
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_GRAPH_DUMPS_DIR) == 0) {
            if (!val.empty()) {
                graph_dumps_dir = val;
                if (mkdir(graph_dumps_dir.c_str(), 0755) != 0) {
                    THROW_IE_EXCEPTION << "Couldn't create clDNN graph dump directory!";
                }
            }
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_SOURCES_DUMPS_DIR) == 0) {
            if (!val.empty()) {
                sources_dumps_dir = val;
                if (mkdir(sources_dumps_dir.c_str(), 0755) != 0) {
                    THROW_IE_EXCEPTION << "Couldn't create clDNN source dump directory!";
                }
            }
        } else if (key.compare(PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                exclusiveAsyncRequests = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                exclusiveAsyncRequests = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS) == 0) {
            if (val.compare(PluginConfigParams::GPU_THROUGHPUT_AUTO) == 0) {
                throughput_streams = 2;
            } else {
                int val_i;
                try {
                    val_i = std::stoi(val);
                } catch (const std::exception&) {
                    THROW_IE_EXCEPTION << "Wrong value for property key " << PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS
                                       << ". Expected only positive numbers (#streams) or "
                                       << "PluginConfigParams::GPU_THROUGHPUT_AUTO";
                }
                if (val_i > 0)
                    throughput_streams = static_cast<uint16_t>(val_i);
            }
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_INT8_ENABLED) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                enableInt8 = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                enableInt8 = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
            }
        } else {
            THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property key by plugin: " << key;
        }

        adjustKeyMapValues();
    }
}

void Config::adjustKeyMapValues() {
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

    if (enableInt8)
        key_config_map[CLDNNConfigParams::KEY_CLDNN_INT8_ENABLED] = PluginConfigParams::YES;
    else
        key_config_map[CLDNNConfigParams::KEY_CLDNN_INT8_ENABLED] = PluginConfigParams::NO;

    {
        std::string qp = "0";
        switch (queuePriority) {
        case cldnn::priority_mode_types::low: qp = "1"; break;
        case cldnn::priority_mode_types::med: qp = "2"; break;
        case cldnn::priority_mode_types::high: qp = "3"; break;
        default: break;
        }
        key_config_map[CLDNNConfigParams::KEY_CLDNN_PLUGIN_PRIORITY] = qp;
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
    }
    {
        std::string tm = PluginConfigParams::TUNING_DISABLED;
        switch (tuningConfig.mode) {
        case cldnn::tuning_mode::tuning_tune_and_cache: tm = PluginConfigParams::TUNING_CREATE; break;
        case cldnn::tuning_mode::tuning_use_cache: tm = PluginConfigParams::TUNING_USE_EXISTING; break;
        default: break;
        }
        key_config_map[PluginConfigParams::KEY_TUNING_MODE] = tm;
        if (!tuningConfig.cache_file_path.empty())
            key_config_map[PluginConfigParams::KEY_TUNING_FILE] = tuningConfig.cache_file_path;
    }

    if (!graph_dumps_dir.empty())
        key_config_map[CLDNNConfigParams::KEY_CLDNN_GRAPH_DUMPS_DIR] = graph_dumps_dir;
    if (!sources_dumps_dir.empty())
        key_config_map[CLDNNConfigParams::KEY_CLDNN_SOURCES_DUMPS_DIR] = sources_dumps_dir;

    key_config_map[PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS] = std::to_string(throughput_streams);
}
}  // namespace CLDNNPlugin
