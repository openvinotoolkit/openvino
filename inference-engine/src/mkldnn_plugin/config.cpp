// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// avoiding clash of the "max" macro with std::max
#define NOMINMAX

#include "config.h"
#include "ie_plugin_config.hpp"
#include "ie_common.h"

#include <string>
#include <cstring>
#include <map>
#include <algorithm>
#include <stdexcept>

#include <cpp_interfaces/exception2status.hpp>
#include <thread>
#include "mkldnn/omp_manager.h"

namespace MKLDNNPlugin {

using namespace InferenceEngine;

void Config::readProperties(const std::map<std::string, std::string> &prop) {
    for (auto& kvp : prop) {
        std::string key = kvp.first;
        std::string val = kvp.second;

        if (key == PluginConfigParams::KEY_CPU_BIND_THREAD) {
            if (val == PluginConfigParams::YES) useThreadBinding = true;
            else if (val == PluginConfigParams::NO) useThreadBinding = false;
            else
                THROW_IE_EXCEPTION << "Wrong value for property key " << PluginConfigParams::KEY_CPU_BIND_THREAD
                                   << ". Expected only YES/NO";
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
                throughputStreams = MKLDNNPlugin::cpu::getNumberOfCPUSockets();
            } else if (val == PluginConfigParams::CPU_THROUGHPUT_AUTO) {
                // bare minimum of streams (that evenly divides available number of core)
                const int num_cores = std::thread::hardware_concurrency();
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
            if (val_i > 0)
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
        } else {
            THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property " << key << " by CPU plugin";
        }
    }
    if (exclusiveAsyncRequests)  // Exclusive request feature disables the streams
        throughputStreams = 1;
}

}  // namespace MKLDNNPlugin
