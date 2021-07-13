// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "threading/ie_istreams_executor.hpp"
#include "ie_plugin_config.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "ie_parallel.hpp"
#include "ie_parallel_custom_arena.hpp"
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
int IStreamsExecutor::Config::GetDefaultNumStreams() {
    const int sockets = static_cast<int>(getAvailableNUMANodes().size());
    // bare minimum of streams (that evenly divides available number of core)
    const int num_cores = sockets == 1 ? std::thread::hardware_concurrency() : getNumberOfCPUCores();
    if (0 == num_cores % 4)
        return std::max(4, num_cores / 4);
    else if (0 == num_cores % 5)
        return std::max(5, num_cores / 5);
    else if (0 == num_cores % 3)
        return std::max(3, num_cores / 3);
    else  // if user disables some cores say in BIOS, so we got weird #cores which is not easy to divide
        return 1;
}

void IStreamsExecutor::Config::SetConfig(const std::string& key, const std::string& value) {
        if (key == CONFIG_KEY(CPU_BIND_THREAD)) {
            if (value == CONFIG_VALUE(YES) || value == CONFIG_VALUE(NUMA)) {
                #if (defined(__APPLE__) || defined(_WIN32))
                _threadBindingType = IStreamsExecutor::ThreadBindingType::NUMA;
                #else
                _threadBindingType = (value == CONFIG_VALUE(YES))
                        ? IStreamsExecutor::ThreadBindingType::CORES : IStreamsExecutor::ThreadBindingType::NUMA;
                #endif
            } else if (value == CONFIG_VALUE(HYBRID_AWARE)) {
                _threadBindingType = IStreamsExecutor::ThreadBindingType::HYBRID_AWARE;
            } else if (value == CONFIG_VALUE(NO)) {
                _threadBindingType = IStreamsExecutor::ThreadBindingType::NONE;
            } else {
                IE_THROW() << "Wrong value for property key " << CONFIG_KEY(CPU_BIND_THREAD)
                                   << ". Expected only YES(binds to cores) / NO(no binding) / NUMA(binds to NUMA nodes) / "
                                                        "HYBRID_AWARE (let the runtime recognize and use the hybrid cores)";
            }
        } else if (key == CONFIG_KEY(CPU_THROUGHPUT_STREAMS)) {
            if (value == CONFIG_VALUE(CPU_THROUGHPUT_NUMA)) {
                _streams = static_cast<int>(getAvailableNUMANodes().size());
            } else if (value == CONFIG_VALUE(CPU_THROUGHPUT_AUTO)) {
                // bare minimum of streams (that evenly divides available number of cores)
                _streams = GetDefaultNumStreams();
            } else {
                int val_i;
                try {
                    val_i = std::stoi(value);
                } catch (const std::exception&) {
                    IE_THROW() << "Wrong value for property key " << CONFIG_KEY(CPU_THROUGHPUT_STREAMS)
                                       << ". Expected only positive numbers (#streams) or "
                                       << "PluginConfigParams::CPU_THROUGHPUT_NUMA/CPU_THROUGHPUT_AUTO";
                }
                if (val_i < 0) {
                    IE_THROW() << "Wrong value for property key " << CONFIG_KEY(CPU_THROUGHPUT_STREAMS)
                                    << ". Expected only positive numbers (#streams)";
                }
                _streams = val_i;
            }
        } else if (key == CONFIG_KEY(CPU_THREADS_NUM)) {
            int val_i;
            try {
                val_i = std::stoi(value);
            } catch (const std::exception&) {
                IE_THROW() << "Wrong value for property key " << CONFIG_KEY(CPU_THREADS_NUM)
                                   << ". Expected only positive numbers (#threads)";
            }
            if (val_i < 0) {
                IE_THROW() << "Wrong value for property key " << CONFIG_KEY(CPU_THREADS_NUM)
                                   << ". Expected only positive numbers (#threads)";
            }
            _threads = val_i;
        } else if (key == CONFIG_KEY_INTERNAL(CPU_THREADS_PER_STREAM)) {
            int val_i;
            try {
                val_i = std::stoi(value);
            } catch (const std::exception&) {
                IE_THROW() << "Wrong value for property key " << CONFIG_KEY_INTERNAL(CPU_THREADS_PER_STREAM)
                                   << ". Expected only non negative numbers (#threads)";
            }
            if (val_i < 0) {
                IE_THROW() << "Wrong value for property key " << CONFIG_KEY_INTERNAL(CPU_THREADS_PER_STREAM)
                                   << ". Expected only non negative numbers (#threads)";
            }
            _threadsPerStream = val_i;
        } else {
            IE_THROW() << "Wrong value for property key " << key;
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
        IE_THROW() << "Wrong value for property key " << key;
    }
    return {};
}

IStreamsExecutor::Config IStreamsExecutor::Config::MakeDefaultMultiThreaded(const IStreamsExecutor::Config& initial, const bool fp_intesive) {
    const auto envThreads = parallel_get_env_threads();
    const auto& numaNodes = getAvailableNUMANodes();
    const int numaNodesNum = numaNodes.size();
    auto streamExecutorConfig = initial;
    const bool bLatencyCase = streamExecutorConfig._streams <= numaNodesNum;

    // by default, do not use the hyper-threading (to minimize threads synch overheads)
    int num_cores_default = getNumberOfCPUCores();
    #if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
    //additional latency-case logic for hybrid processors:
    if (ThreadBindingType::HYBRID_AWARE == streamExecutorConfig._threadBindingType) {
        const auto core_types = custom::info::core_types();
        const auto num_little_cores = custom::info::default_concurrency(custom::task_arena::constraints{}.set_core_type(core_types.front()));
        const auto num_big_cores_phys = getNumberOfCPUCores(true);
        const int int8_threshold = 4; // ~relative efficiency of the VNNI-intensive code for Big vs Little cores;
        const int fp32_threshold = 2; // ~relative efficiency of the AVX2 fp32 code for Big vs Little cores;
        // by default the latency case uses (faster) Big cores only, depending on the compute ratio
        const bool bLatencyCaseBigOnly = num_big_cores_phys > (num_little_cores / (fp_intesive ? fp32_threshold : int8_threshold));
        // selecting the preferred core type
        streamExecutorConfig._threadPreferredCoreType =
            bLatencyCase
                ? (bLatencyCaseBigOnly
                    ? IStreamsExecutor::Config::PreferredCoreType::BIG
                    : IStreamsExecutor::Config::PreferredCoreType::ANY)
                : IStreamsExecutor::Config::PreferredCoreType::ROUND_ROBIN;
        // additionally selecting the #cores to use in the "Big-only" case
        if (bLatencyCaseBigOnly) {
            const int hyper_threading_threshold = 2; // min #cores, for which the hyper-threading becomes useful for the latency case
            const auto num_big_cores = custom::info::default_concurrency(custom::task_arena::constraints{}.set_core_type(core_types.back()));
            num_cores_default = (num_big_cores_phys <= hyper_threading_threshold) ? num_big_cores : num_big_cores_phys;
        }
    }
    #endif
    const auto hwCores = !bLatencyCase && numaNodesNum == 1
        // throughput case on a single-NUMA node machine uses all available cores
        ? parallel_get_max_threads()
        // in the rest of cases:
        //    multi-node machine
        //    or
        //    latency case, single-node yet hybrid case that uses
        //      all core types
        //      or
        //      big-cores only, but the #cores is "enough" (pls see the logic above)
        // it is usually beneficial not to use the hyper-threading (which is default)
        : num_cores_default;
    const auto threads = streamExecutorConfig._threads ? streamExecutorConfig._threads : (envThreads ? envThreads : hwCores);
    streamExecutorConfig._threadsPerStream = streamExecutorConfig._streams
                                            ? std::max(1, threads/streamExecutorConfig._streams)
                                            : threads;
    return streamExecutorConfig;
}

}  //  namespace InferenceEngine
