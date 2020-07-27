// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "threading/ie_istreams_executor.hpp"
#include "ie_plugin_config.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "details/ie_exception.hpp"
#include "ie_parallel.hpp"
#include "ie_system_conf.h"
#include "ie_parameter.hpp"
#include <string>
#include <algorithm>
#include <vector>
#include <thread>


namespace InferenceEngine {
IStreamsExecutor::~IStreamsExecutor() {}

std::vector<std::string> IStreamsExecutor::Config::SupportedKeys() {
    return {
        CONFIG_KEY(CPU_THROUGHPUT_STREAMS),
        CONFIG_KEY(CPU_BIND_THREAD),
        CONFIG_KEY(CPU_THREADS_NUM),
        CONFIG_KEY_INTERNAL(CPU_THREADS_PER_STREAM),
    };
}

void IStreamsExecutor::Config::SetConfig(const std::string& key, const std::string& value) {
        if (key == CONFIG_KEY(CPU_BIND_THREAD)) {
            if (value == CONFIG_VALUE(YES) || value == CONFIG_VALUE(NUMA)) {
#if (defined(__APPLE__) || defined(_WIN32))
                // on the Windows and Apple the CORES and NUMA pinning options are the same
                _threadBindingType = IStreamsExecutor::ThreadBindingType::NUMA;
#else
                _threadBindingType = (value == CONFIG_VALUE(YES))
                        ? IStreamsExecutor::ThreadBindingType::CORES : IStreamsExecutor::ThreadBindingType::NUMA;
#endif
            } else if (value == CONFIG_VALUE(NO)) {
                _threadBindingType = IStreamsExecutor::ThreadBindingType::NONE;
            } else {
                THROW_IE_EXCEPTION << "Wrong value for property key " << CONFIG_KEY(CPU_BIND_THREAD)
                                   << ". Expected only YES(binds to cores) / NO(no binding) / NUMA(binds to NUMA nodes)";
            }
        } else if (key == CONFIG_KEY(CPU_THROUGHPUT_STREAMS)) {
            if (value == CONFIG_VALUE(CPU_THROUGHPUT_NUMA)) {
                _streams = getAvailableNUMANodes().size();
            } else if (value == CONFIG_VALUE(CPU_THROUGHPUT_AUTO)) {
                const int sockets = getAvailableNUMANodes().size();
                // bare minimum of streams (that evenly divides available number of core)
                const int num_cores = sockets == 1 ? std::thread::hardware_concurrency() : getNumberOfCPUCores();
                if (0 == num_cores % 4)
                    _streams = std::max(4, num_cores / 4);
                else if (0 == num_cores % 5)
                    _streams = std::max(5, num_cores / 5);
                else if (0 == num_cores % 3)
                    _streams = std::max(3, num_cores / 3);
                else  // if user disables some cores say in BIOS, so we got weird #cores which is not easy to divide
                    _streams = 1;
            } else {
                int val_i;
                try {
                    val_i = std::stoi(value);
                } catch (const std::exception&) {
                    THROW_IE_EXCEPTION << "Wrong value for property key " << CONFIG_KEY(CPU_THROUGHPUT_STREAMS)
                                       << ". Expected only positive numbers (#streams) or "
                                       << "PluginConfigParams::CPU_THROUGHPUT_NUMA/CPU_THROUGHPUT_AUTO";
                }
                if (val_i > 0)
                    _streams = val_i;
            }
        } else if (key == CONFIG_KEY(CPU_THREADS_NUM)) {
            int val_i;
            try {
                val_i = std::stoi(value);
            } catch (const std::exception&) {
                THROW_IE_EXCEPTION << "Wrong value for property key " << CONFIG_KEY(CPU_THREADS_NUM)
                                   << ". Expected only positive numbers (#threads)";
            }
            if (val_i <= 0) {
                THROW_IE_EXCEPTION << "Wrong value for property key " << CONFIG_KEY(CPU_THREADS_NUM)
                                   << ". Expected only positive numbers (#threads)";
            }
            _threads = val_i;
        } else if (key == CONFIG_KEY_INTERNAL(CPU_THREADS_PER_STREAM)) {
            int val_i;
            try {
                val_i = std::stoi(value);
            } catch (const std::exception&) {
                THROW_IE_EXCEPTION << "Wrong value for property key " << CONFIG_KEY_INTERNAL(CPU_THREADS_PER_STREAM)
                                   << ". Expected only non negative numbers (#threads)";
            }
            if (val_i < 0) {
                THROW_IE_EXCEPTION << "Wrong value for property key " << CONFIG_KEY_INTERNAL(CPU_THREADS_PER_STREAM)
                                   << ". Expected only non negative numbers (#threads)";
            }
            _threadsPerStream = val_i;
        } else {
            THROW_IE_EXCEPTION << "Wrong value for property key " << key;
        }
}

Parameter IStreamsExecutor::Config::GetConfig(const std::string& key) {
    if (key == CONFIG_KEY(CPU_BIND_THREAD)) {
        switch (_threadBindingType) {
            case IStreamsExecutor::ThreadBindingType::NONE:
                return {CONFIG_VALUE(NO)};
            break;
            case IStreamsExecutor::ThreadBindingType::CORES:
                return {CONFIG_VALUE(YES)};
            break;
            case IStreamsExecutor::ThreadBindingType::NUMA:
                return {CONFIG_VALUE(NUMA)};
            break;
        }
    } else if (key == CONFIG_KEY(CPU_THROUGHPUT_STREAMS)) {
        return {_streams};
    } else if (key == CONFIG_KEY(CPU_THREADS_NUM)) {
        return {_threads};
    } else if (key == CONFIG_KEY_INTERNAL(CPU_THREADS_PER_STREAM)) {
        return {_threadsPerStream};
    } else {
        THROW_IE_EXCEPTION << "Wrong value for property key " << key;
    }
    return {};
}

IStreamsExecutor::Config IStreamsExecutor::Config::MakeDefaultMultiThreaded(const IStreamsExecutor::Config& initial) {
    const auto envThreads = parallel_get_env_threads();
    const auto& numaNodes = getAvailableNUMANodes();
    const auto numaNodesNum = numaNodes.size();
    auto streamExecutorConfig = initial;
    const auto hwCores = streamExecutorConfig._streams > 1 && numaNodesNum == 1 ? parallel_get_max_threads() : getNumberOfCPUCores();
    const auto threads = streamExecutorConfig._threads ? streamExecutorConfig._threads : (envThreads ? envThreads : hwCores);
    streamExecutorConfig._threadsPerStream = streamExecutorConfig._streams
                                            ? std::max(1, threads/streamExecutorConfig._streams)
                                            : threads;
    return streamExecutorConfig;
}

}  //  namespace InferenceEngine