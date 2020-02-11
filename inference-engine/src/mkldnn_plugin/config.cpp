// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// avoiding clash of the "max" macro with std::max
#define NOMINMAX

#include "config.h"

#include <string>
#include <map>
#include <algorithm>

#include "ie_plugin_config.hpp"
#include "ie_common.h"

#include <cpp_interfaces/exception2status.hpp>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include <ie_parallel.hpp>
#include "mkldnn/system_conf.h"

namespace MKLDNNPlugin {

using namespace InferenceEngine;

void Config::readProperties(const std::map<std::string, std::string> &prop) {
    for (auto& kvp : prop) {
        std::string key = kvp.first;
        std::string val = kvp.second;

        if (key == PluginConfigParams::KEY_CPU_BIND_THREAD) {
            if (val == PluginConfigParams::YES || val == PluginConfigParams::NUMA) {
                #ifdef _WIN32
                // on the Windows the CORES and NUMA pinning options are the same
                useThreadBinding = InferenceThreadsBinding::NUMA;
                #else
                useThreadBinding = (val == PluginConfigParams::YES)
                        ? InferenceThreadsBinding::CORES : InferenceThreadsBinding::NUMA;
                #endif
            } else if (val == PluginConfigParams::NO) {
                useThreadBinding = InferenceThreadsBinding::NONE;
            } else {
                THROW_IE_EXCEPTION << "Wrong value for property key " << PluginConfigParams::KEY_CPU_BIND_THREAD
                                   << ". Expected only YES(binds to cores) / NO(no binding) / NUMA(binds to NUMA nodes)";
            }
        } else if (key == PluginConfigParams::KEY_DYN_BATCH_LIMIT) {
            int val_i = std::stoi(val);
            // zero and any negative value will be treated
            // as default batch size
            batchLimit = std::max(val_i, 0);
        } else if (key == PluginConfigParams::KEY_PERF_COUNT) {
            if (val == PluginConfigParams::YES) collectPerfCounters = true;
            else if (val == PluginConfigParams::NO) collectPerfCounters = false;
            else
                THROW_IE_EXCEPTION << "Wrong value for property key " << PluginConfigParams::KEY_PERF_COUNT
                                   << ". Expected only YES/NO";
        } else if (key == PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS) {
            if (val == PluginConfigParams::YES) exclusiveAsyncRequests = true;
            else if (val == PluginConfigParams::NO) exclusiveAsyncRequests = false;
            else
                THROW_IE_EXCEPTION << "Wrong value for property key " << PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS
                                   << ". Expected only YES/NO";
        } else if (key == PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS) {
            if (val == PluginConfigParams::CPU_THROUGHPUT_NUMA) {
                throughputStreams = MKLDNNPlugin::cpu::getAvailableNUMANodes().size();
            } else if (val == PluginConfigParams::CPU_THROUGHPUT_AUTO) {
                const int sockets = MKLDNNPlugin::cpu::getAvailableNUMANodes().size();
                // bare minimum of streams (that evenly divides available number of core)
                const int num_cores = sockets == 1 ? parallel_get_max_threads() : cpu::getNumberOfCPUCores();
                if (0 == num_cores % 4)
                    throughputStreams = std::max(4, num_cores / 4);
                else if (0 == num_cores % 5)
                    throughputStreams = std::max(5, num_cores / 5);
                else if (0 == num_cores % 3)
                    throughputStreams = std::max(3, num_cores / 3);
                else  // if user disables some cores say in BIOS, so we got weird #cores which is not easy to divide
                    throughputStreams = 1;
            } else {
                int val_i;
                try {
                    val_i = std::stoi(val);
                } catch (const std::exception&) {
                    THROW_IE_EXCEPTION << "Wrong value for property key " << PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS
                                       << ". Expected only positive numbers (#streams) or "
                                       << "PluginConfigParams::CPU_THROUGHPUT_NUMA/CPU_THROUGHPUT_AUTO";
                }
                if (val_i > 0)
                    throughputStreams = val_i;
            }
        } else if (key == PluginConfigParams::KEY_CPU_THREADS_NUM) {
            int val_i;
            try {
                val_i = std::stoi(val);
            } catch (const std::exception&) {
                THROW_IE_EXCEPTION << "Wrong value for property key " << PluginConfigParams::KEY_CPU_THREADS_NUM
                                   << ". Expected only positive numbers (#threads)";
            }
            if (val_i < 0)
                THROW_IE_EXCEPTION << "Wrong value for property key " << PluginConfigParams::KEY_CPU_THREADS_NUM
                                   << ". Expected only positive numbers (#threads)";
            threadsNum = val_i;
        } else if (key.compare(PluginConfigParams::KEY_DYN_BATCH_ENABLED) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0)
                enableDynamicBatch = true;
            else if (val.compare(PluginConfigParams::NO) == 0)
                enableDynamicBatch = false;
            else
                THROW_IE_EXCEPTION << "Wrong value for property key " << PluginConfigParams::KEY_DYN_BATCH_ENABLED
                << ". Expected only YES/NO";
        } else if (key.compare(PluginConfigParams::KEY_DUMP_EXEC_GRAPH_AS_DOT) == 0) {
            // empty string means that dumping is switched off
            dumpToDot = val;
        } else if (key.compare(PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE) == 0) {
            if (val == PluginConfigInternalParams::LP_TRANSFORMS_MODE_OFF)
                lpTransformsMode = LPTransformsMode::Off;
            else if (val == PluginConfigInternalParams::LP_TRANSFORMS_MODE_ON)
                lpTransformsMode = LPTransformsMode::On;
            else
                THROW_IE_EXCEPTION << "Wrong value for property key " << PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE;
        } else if (key.compare(PluginConfigParams::KEY_DUMP_QUANTIZED_GRAPH_AS_DOT) == 0) {
            dumpQuantizedGraphToDot = val;
        } else if (key.compare(PluginConfigParams::KEY_DUMP_QUANTIZED_GRAPH_AS_IR) == 0) {
            dumpQuantizedGraphToIr = val;
        } else {
            THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property " << key << " by CPU plugin";
        }
        _config.clear();
    }
    if (exclusiveAsyncRequests)  // Exclusive request feature disables the streams
        throughputStreams = 1;

    updateProperties();
}
void Config::updateProperties() {
    if (!_config.size()) {
        if (useThreadBinding == true)
            _config.insert({ PluginConfigParams::KEY_CPU_BIND_THREAD, PluginConfigParams::YES });
        else
            _config.insert({ PluginConfigParams::KEY_CPU_BIND_THREAD, PluginConfigParams::NO });
        if (collectPerfCounters == true)
            _config.insert({ PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES });
        else
            _config.insert({ PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::NO });
        if (exclusiveAsyncRequests == true)
            _config.insert({ PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, PluginConfigParams::YES });
        else
            _config.insert({ PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, PluginConfigParams::NO });
        if (enableDynamicBatch == true)
            _config.insert({ PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES });
        else
            _config.insert({ PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::NO });

        _config.insert({ PluginConfigParams::KEY_DYN_BATCH_LIMIT, std::to_string(batchLimit) });
        _config.insert({ PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, std::to_string(throughputStreams) });
        _config.insert({ PluginConfigParams::KEY_CPU_THREADS_NUM, std::to_string(threadsNum) });
        _config.insert({ PluginConfigParams::KEY_DUMP_EXEC_GRAPH_AS_DOT, dumpToDot });
    }
}

}  // namespace MKLDNNPlugin
