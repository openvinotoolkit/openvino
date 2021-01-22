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
#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO) && (TBB_INTERFACE_VERSION < 11100)
                if (value == CONFIG_VALUE(NUMA))
                    THROW_IE_EXCEPTION << CONFIG_KEY(CPU_BIND_THREAD) << " property value was set to NUMA. But IE was built with "
                                       << "TBB version without NUMA-aware API. Current TBB API version is " << TBB_INTERFACE_VERSION
                                       << ", required API version 11100 or greater.";
#endif

#if (defined(__APPLE__) || defined(_WIN32))
                _threadBindingType = IStreamsExecutor::ThreadBindingType::NUMA;
#else
                _threadBindingType = (value == CONFIG_VALUE(YES))
                        ? IStreamsExecutor::ThreadBindingType::CORES : IStreamsExecutor::ThreadBindingType::NUMA;
#endif
            } else if (value == CONFIG_VALUE(HYBRID_AWARE)) {
#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO) && (TBB_INTERFACE_VERSION < 12010)
                THROW_IE_EXCEPTION << CONFIG_KEY(CPU_BIND_THREAD) << " property value was set to HYBRID_AWARE. But IE was built with "
                << "TBB version without Hybrid-aware API. Current TBB API version is " << TBB_INTERFACE_VERSION
                << ", required API version 12010 or greater.";
#endif
                _threadBindingType = IStreamsExecutor::ThreadBindingType::HYBRID_AWARE;
            } else if (value == CONFIG_VALUE(NO)) {
                _threadBindingType = IStreamsExecutor::ThreadBindingType::NONE;
            } else {
                THROW_IE_EXCEPTION << "Wrong value for property key " << CONFIG_KEY(CPU_BIND_THREAD)
                                   << ". Expected only YES(binds to cores) / NO(no binding) / NUMA(binds to NUMA nodes) / "
                                                        "HYBRID_AWARE (let the runtime recognize and use the hybrid cores)";
            }
        } else if (key == CONFIG_KEY(CPU_THROUGHPUT_STREAMS)) {
            if (value == CONFIG_VALUE(CPU_THROUGHPUT_NUMA)) {
                _streams = static_cast<int>(getAvailableNUMANodes().size());
            } else if (value == CONFIG_VALUE(CPU_THROUGHPUT_AUTO)) {
                const int sockets = static_cast<int>(getAvailableNUMANodes().size());
                // bare minimum of streams (that evenly divides available number of cores)
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
                if (val_i < 0) {
                    THROW_IE_EXCEPTION << "Wrong value for property key " << CONFIG_KEY(CPU_THROUGHPUT_STREAMS)
                                    << ". Expected only positive numbers (#streams)";
                }
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
            if (val_i < 0) {
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
            case IStreamsExecutor::ThreadBindingType::HYBRID_AWARE:
                return {CONFIG_VALUE(HYBRID_AWARE)};
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
    const bool bLatencyCase = streamExecutorConfig._streams <= numaNodesNum;
    bool bLatencyCaseHybrid = false;
    //additional latency-case logic for hybrid processors:
    #if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO) && (TBB_INTERFACE_VERSION >= 12010) // TBB with hybrid CPU aware task_arena api
    if (ThreadBindingType::HYBRID_AWARE == streamExecutorConfig._threadBindingType) {
        const auto core_types = oneapi::tbb::info::core_types();
        const auto num_strongest_cores = oneapi::tbb::info::default_concurrency(core_types.back());
        const auto num_other_cores = std::accumulate(core_types.cbegin(), core_types.cend(), -num_strongest_cores /*without big cores*/,
            [](const auto& p, int num) {return num + oneapi::tbb::info::default_concurrency(p); });
        // by default the latency case uses (faster) Big cores only
        bLatencyCaseHybrid = num_strongest_cores > num_other_cores / 4; // relative efficiency of the VNNI-intensive code for Big vs Little cores;
        streamExecutorConfig._threadPreferredCoreType =
                                bLatencyCase
                                    ? (bLatencyCaseHybrid                                       
                                        ? IStreamsExecutor::Config::PreferredCoreType::BIG
                                        : IStreamsExecutor::Config::PreferredCoreType::NONE)
                                    : IStreamsExecutor::Config::PreferredCoreType::ROUND_ROBIN;
    }
    #endif

    const auto hwCores = !bLatencyCase && numaNodesNum == 1
        // throughput case on a single-NUMA node machine uses all available cores
        ? parallel_get_max_threads()
        // on multi-NUMA node do not use hyper-threading (to reduce memory pressure)
        // additionally, any latency case (stream per node, or less) runs on the Big cores only
        : getNumberOfCPUCores(bLatencyCaseHybrid);
    const auto threads = streamExecutorConfig._threads ? streamExecutorConfig._threads : (envThreads ? envThreads : hwCores);
    streamExecutorConfig._threadsPerStream = streamExecutorConfig._streams
                                            ? std::max(1, threads/streamExecutorConfig._streams)
                                            : threads;
    return streamExecutorConfig;
}

}  //  namespace InferenceEngine
