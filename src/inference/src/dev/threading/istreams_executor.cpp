// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/threading/istreams_executor.hpp"

#include <algorithm>
#include <string>
#include <thread>
#include <vector>

#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "ie_plugin_config.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/property_supervisor.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/util/log.hpp"
#include "threading/ie_parallel_custom_arena.hpp"

namespace ov {
namespace threading {

IStreamsExecutor::~IStreamsExecutor() {}

IStreamsExecutor::Config::Config(const std::string& name, const ov::AnyMap& config) : m_name(name) {
    // Add default properties
    m_properties
        .add(ov::threading::IStreamsExecutor::Config::name, std::ref(m_name))

        .add(ov::threading::IStreamsExecutor::Config::streams,
             std::ref(m_streams),
             [](const int32_t& streams) {
                 OPENVINO_ASSERT(streams >= 0);
             })
        .add(ov::threading::IStreamsExecutor::Config::threads_per_stream,
             std::ref(m_threads_per_stream),
             [](const int32_t& threads) {
                 OPENVINO_ASSERT(threads >= 0);
             })
        .add(ov::threading::IStreamsExecutor::Config::thread_binding_type,
             std::ref(m_thread_binding_type),
             [](const int32_t& threads) {
                 OPENVINO_ASSERT(threads >= 0);
             })
        .add(ov::threading::IStreamsExecutor::Config::thread_binding_step,
             std::ref(m_thread_binding_step),
             [](const int32_t& threads) {
                 OPENVINO_ASSERT(threads >= 0);
             })
        .add(ov::threading::IStreamsExecutor::Config::thread_binding_offset,
             std::ref(m_thread_binding_offset),
             [](const int32_t& offset) {
                 OPENVINO_ASSERT(offset >= 0);
             })
        .add(ov::threading::IStreamsExecutor::Config::threads,
             std::ref(m_threads),
             [](const int32_t& threads) {
                 OPENVINO_ASSERT(threads >= 0);
             })
        .add(ov::threading::IStreamsExecutor::Config::big_core_streams,
             std::ref(m_big_core_streams),
             [](const int32_t& streams) {
                 OPENVINO_ASSERT(streams >= 0);
             })
        .add(ov::threading::IStreamsExecutor::Config::small_core_streams,
             std::ref(m_small_core_streams),
             [](const int32_t& streams) {
                 OPENVINO_ASSERT(streams >= 0);
             })
        .add(ov::threading::IStreamsExecutor::Config::threads_per_stream_big,
             std::ref(m_threads_per_stream_big),
             [](const int32_t& streams) {
                 OPENVINO_ASSERT(streams >= 0);
             })
        .add(ov::threading::IStreamsExecutor::Config::threads_per_stream_small,
             std::ref(m_threads_per_stream_small),
             [](const int32_t& streams) {
                 OPENVINO_ASSERT(streams >= 0);
             })
        .add(ov::threading::IStreamsExecutor::Config::small_core_offset,
             std::ref(m_small_core_offset),
             [](const int32_t& streams) {
                 OPENVINO_ASSERT(streams >= 0);
             })
        .add(ov::threading::IStreamsExecutor::Config::enable_hyper_thread, std::ref(m_enable_hyper_thread))
        .add(ov::threading::IStreamsExecutor::Config::thread_preferred_core_type,
             std::ref(m_thread_preferred_core_type));

    // Add some legacy properties
    m_properties
        .add(
            ov::legacy_property(CONFIG_KEY(CPU_BIND_THREAD)),
            [this]() -> std::string {
                switch (m_thread_binding_type) {
                case IStreamsExecutor::ThreadBindingType::NONE:
                    return {CONFIG_VALUE(NO)};
                case IStreamsExecutor::ThreadBindingType::CORES:
                    return {CONFIG_VALUE(YES)};
                case IStreamsExecutor::ThreadBindingType::NUMA:
                    return {CONFIG_VALUE(NUMA)};
                case IStreamsExecutor::ThreadBindingType::HYBRID_AWARE:
                    return {CONFIG_VALUE(HYBRID_AWARE)};
                default:
                    OPENVINO_UNREACHABLE("Unsupported thread binding type");
                }
            },
            [this](const std::string& value) {
                if (value == CONFIG_VALUE(YES) || value == CONFIG_VALUE(NUMA)) {
#if (defined(__APPLE__) || defined(_WIN32))
                    m_thread_binding_type = IStreamsExecutor::ThreadBindingType::NUMA;
#else
      m_thread_binding_type = (value == CONFIG_VALUE(YES)) ? IStreamsExecutor::ThreadBindingType::CORES
                : IStreamsExecutor::ThreadBindingType::NUMA;
#endif
                } else if (value == CONFIG_VALUE(HYBRID_AWARE)) {
                    m_thread_binding_type = IStreamsExecutor::ThreadBindingType::HYBRID_AWARE;
                } else if (value == CONFIG_VALUE(NO)) {
                    m_thread_binding_type = IStreamsExecutor::ThreadBindingType::NONE;
                } else {
                    IE_THROW() << "Wrong value for property key " << CONFIG_KEY(CPU_BIND_THREAD);
                }
            })
        .add(
            ov::legacy_property(CONFIG_KEY(CPU_THROUGHPUT_STREAMS)),
            [this] {
                return std::to_string(m_streams);
            },
            [this](const std::string& value) {
                if (value == CONFIG_VALUE(CPU_THROUGHPUT_NUMA)) {
                    m_streams = static_cast<int32_t>(get_available_numa_nodes().size());
                } else if (value == CONFIG_VALUE(CPU_THROUGHPUT_AUTO)) {
                    // bare minimum of streams (that evenly divides available number of cores)
                    m_streams = get_default_num_streams();
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
                    m_streams = val_i;
                }
            })
        .add(ov::internal_property(CONFIG_KEY_INTERNAL(CPU_THREADS_PER_STREAM)),
             std::ref(m_threads_per_stream),
             [](const int32_t& threads) {
                 OPENVINO_ASSERT(threads >= 0);
             })
        .add(ov::internal_property(CONFIG_KEY_INTERNAL(BIG_CORE_STREAMS)),
             std::ref(m_big_core_streams),
             [](const int32_t& streams) {
                 OPENVINO_ASSERT(streams >= 0);
             })
        .add(ov::internal_property(CONFIG_KEY_INTERNAL(THREADS_PER_STREAM_BIG)),
             std::ref(m_threads_per_stream_big),
             [](const int32_t& threads) {
                 OPENVINO_ASSERT(threads >= 0);
             })
        .add(ov::internal_property(CONFIG_KEY_INTERNAL(SMALL_CORE_STREAMS)),
             std::ref(m_small_core_streams),
             [](const int32_t& streams) {
                 OPENVINO_ASSERT(streams >= 0);
             })
        .add(ov::internal_property(CONFIG_KEY_INTERNAL(THREADS_PER_STREAM_SMALL)),
             std::ref(m_threads_per_stream_small),
             [](const int32_t& threads) {
                 OPENVINO_ASSERT(threads >= 0);
             })
        .add(ov::internal_property(CONFIG_KEY_INTERNAL(SMALL_CORE_OFFSET)),
             std::ref(m_small_core_offset),
             [](const int32_t& offset) {
                 OPENVINO_ASSERT(offset >= 0);
             })
        .add(ov::internal_property(CONFIG_KEY_INTERNAL(ENABLE_HYPER_THREAD)), std::ref(m_enable_hyper_thread))
        .add(
            ov::legacy_property(CONFIG_KEY(CPU_THREADS_NUM)),
            [this] {
                return std::to_string(m_threads);
            },
            [this](const std::string& value) {
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
                m_threads = val_i;
            })
        .add(
            ov::streams::num,
            [this] {
                return m_streams;
            },
            [this](const int32_t& streams) {
                if (streams == ov::streams::NUMA) {
                    m_streams = static_cast<int32_t>(get_available_numa_nodes().size());
                } else if (streams == ov::streams::AUTO) {
                    // bare minimum of streams (that evenly divides available number of cores)
                    m_streams = get_default_num_streams();
                } else {
                    m_streams = streams;
                }
            },
            [](const int32_t& streams) {
                OPENVINO_ASSERT((streams == ov::streams::NUMA) || (streams == ov::streams::AUTO) || (streams >= 0),
                                "Wrong value for property key ",
                                ov::streams::num.name(),
                                ". Expected non negative numbers (#streams) or ",
                                "ov::streams::NUMA|ov::streams::AUTO: ",
                                streams);
            })
        .add(ov::inference_num_threads,
             std::ref(m_threads),
             [](const int32_t& threads) {
                 OPENVINO_ASSERT(threads >= 0);
             })
        .add(
            ov::affinity,
            [this] {
                switch (m_thread_binding_type) {
                case ThreadBindingType::NONE:
                    return ov::Affinity::NONE;
                case ThreadBindingType::CORES:
                    return ov::Affinity::CORE;
                case ThreadBindingType::NUMA:
                    return ov::Affinity::NUMA;
                case ThreadBindingType::HYBRID_AWARE:
                    return ov::Affinity::HYBRID_AWARE;
                default:
                    OPENVINO_UNREACHABLE("Unsupported thread binding type");
                };
            },
            [this](const ov::Affinity& affinity) {
                switch (affinity) {
                case ov::Affinity::NONE:
                    m_thread_binding_type = ThreadBindingType::NONE;
                    break;
                case ov::Affinity::CORE: {
#if (defined(__APPLE__) || defined(_WIN32))
                    m_thread_binding_type = ThreadBindingType::NUMA;
#else
                          m_thread_binding_type = ThreadBindingType::CORES;
#endif
                    break;
                }
                case ov::Affinity::NUMA:
                    m_thread_binding_type = ThreadBindingType::NUMA;
                    break;
                case ov::Affinity::HYBRID_AWARE:
                    m_thread_binding_type = ThreadBindingType::HYBRID_AWARE;
                    break;
                default:
                    OPENVINO_UNREACHABLE("Unsupported affinity type");
                };
            });
    m_properties.set(config, true);
}

void IStreamsExecutor::Config::set_property(const std::string& key, const ov::Any& value) {
    set_property({{key, value}});
}

void IStreamsExecutor::Config::set_property(const ov::AnyMap& property) {
    m_properties.set(property);
}

ov::Any IStreamsExecutor::Config::get_property(const std::string& key) const {
    return m_properties.get(key);
}

int IStreamsExecutor::Config::get_default_num_streams(const bool enable_hyper_thread) {
    const int sockets = static_cast<int>(get_available_numa_nodes().size());
    // bare minimum of streams (that evenly divides available number of core)
    const int num_cores = sockets == 1 ? (enable_hyper_thread ? parallel_get_max_threads() : get_number_of_cpu_cores())
                                       : get_number_of_cpu_cores();
    if (0 == num_cores % 4)
        return std::max(4, num_cores / 4);
    else if (0 == num_cores % 5)
        return std::max(5, num_cores / 5);
    else if (0 == num_cores % 3)
        return std::max(3, num_cores / 3);
    else  // if user disables some cores say in BIOS, so we got weird #cores which is not easy to divide
        return 1;
}

int IStreamsExecutor::Config::get_hybrid_num_streams(std::map<std::string, std::string>& config,
                                                     const int stream_mode) {
    const int num_cores = parallel_get_max_threads();
    const int num_cores_phy = get_number_of_cpu_cores();
    const int num_big_cores_phy = get_number_of_cpu_cores(true);
    const int num_small_cores = num_cores_phy - num_big_cores_phy;
    const int num_big_cores = num_cores > num_cores_phy ? num_big_cores_phy * 2 : num_big_cores_phy;
    int big_core_streams = 0;
    int small_core_streams = 0;
    int threads_per_stream_big = 0;
    int threads_per_stream_small = 0;

    if (stream_mode == DEFAULT) {
        // bare minimum of streams (that evenly divides available number of core)
        if (0 == num_big_cores_phy % 4) {
            threads_per_stream_big = 4;
        } else if (0 == num_big_cores_phy % 5) {
            threads_per_stream_big = 5;
        } else if (0 == num_big_cores_phy % 3) {
            threads_per_stream_big = 3;
        } else {  // if user disables some cores say in BIOS, so we got weird #cores which is not easy to divide
            threads_per_stream_big = num_big_cores_phy;
        }

        big_core_streams = num_big_cores / threads_per_stream_big;
        threads_per_stream_small = threads_per_stream_big;
        if (num_small_cores == 0) {
            threads_per_stream_small = 0;
        } else if (num_small_cores < threads_per_stream_small) {
            small_core_streams = 1;
            threads_per_stream_small = num_small_cores;
            threads_per_stream_big = threads_per_stream_small;
            // Balance the computation of physical core and logical core, the number of threads on the physical core and
            // logical core should be equal
            big_core_streams = num_big_cores_phy / threads_per_stream_big * 2;
        } else {
            small_core_streams = num_small_cores / threads_per_stream_small;
        }
    } else if (stream_mode == AGGRESSIVE) {
        big_core_streams = num_big_cores;
        small_core_streams = num_small_cores;
        threads_per_stream_big = num_big_cores / big_core_streams;
        threads_per_stream_small = num_small_cores == 0 ? 0 : num_small_cores / small_core_streams;
    } else if (stream_mode == LESSAGGRESSIVE) {
        big_core_streams = num_big_cores / 2;
        small_core_streams = num_small_cores / 2;
        threads_per_stream_big = num_big_cores / big_core_streams;
        threads_per_stream_small = num_small_cores == 0 ? 0 : num_small_cores / small_core_streams;
    } else {
        IE_THROW() << "Wrong stream mode to get num of streams: " << stream_mode;
    }
    config[CONFIG_KEY_INTERNAL(BIG_CORE_STREAMS)] = std::to_string(big_core_streams);
    config[CONFIG_KEY_INTERNAL(SMALL_CORE_STREAMS)] = std::to_string(small_core_streams);
    config[CONFIG_KEY_INTERNAL(THREADS_PER_STREAM_BIG)] = std::to_string(threads_per_stream_big);
    config[CONFIG_KEY_INTERNAL(THREADS_PER_STREAM_SMALL)] = std::to_string(threads_per_stream_small);
    // This is default setting for specific CPU which Pcore is in front and Ecore is in the back.
    config[CONFIG_KEY_INTERNAL(SMALL_CORE_OFFSET)] = std::to_string(num_small_cores == 0 ? 0 : num_big_cores);
    return big_core_streams + small_core_streams;
}

void IStreamsExecutor::Config::update_hybrid_custom_threads(Config& config) {
    const auto num_cores = parallel_get_max_threads();
    const auto num_cores_phys = get_number_of_cpu_cores();
    const auto num_big_cores_phys = get_number_of_cpu_cores(true);
    const auto num_big_cores = num_cores > num_cores_phys ? num_big_cores_phys * 2 : num_big_cores_phys;
    const auto num_small_cores_phys = num_cores_phys - num_big_cores_phys;
    const auto threads = config.get_property(Config::threads.name()).as<int32_t>()
                             ? config.get_property(Config::threads.name()).as<int32_t>()
                             : num_cores;
    const auto streams = config.get_property(Config::streams.name()).as<int32_t>() > 0
                             ? config.get_property(Config::streams.name()).as<int32_t>()
                             : 1;

    config.set_property(Config::small_core_offset.name(), num_big_cores);
    int threads_per_stream = std::max(1, threads / streams);

    if ((num_big_cores_phys / threads_per_stream >= streams) && (1 < threads_per_stream)) {
        config.set_property(Config::big_core_streams.name(), streams);
        config.set_property(Config::threads_per_stream_big.name(), threads_per_stream);
        config.set_property(Config::small_core_streams.name(), 0);
        config.set_property(Config::threads_per_stream_small.name(), 0);
    } else if ((num_small_cores_phys / threads_per_stream >= streams) && (num_big_cores_phys < threads_per_stream)) {
        config.set_property(Config::big_core_streams.name(), 0);
        config.set_property(Config::threads_per_stream_big.name(), 0);
        config.set_property(Config::small_core_streams.name(), streams);
        config.set_property(Config::threads_per_stream_small.name(), threads_per_stream);
    } else {
        const int threads_per_stream_big = std::min(num_big_cores_phys, threads_per_stream);
        const int threads_per_stream_small = std::min(num_small_cores_phys, threads_per_stream);

        threads_per_stream = std::min(threads_per_stream_big, threads_per_stream_small);
        while (threads_per_stream > 1) {
            const int base_big_streams = num_big_cores_phys / threads_per_stream;
            const int base_small_streams = num_small_cores_phys > 0 ? num_small_cores_phys / threads_per_stream : 0;
            if (base_big_streams + base_small_streams >= streams) {
                config.set_property(Config::big_core_streams.name(), base_big_streams);
                config.set_property(Config::small_core_streams.name(), streams - base_big_streams);
                break;
            } else if (base_big_streams * 2 + base_small_streams >= streams) {
                config.set_property(Config::big_core_streams.name(), streams - base_small_streams);
                config.set_property(Config::small_core_streams.name(), base_small_streams);
                break;
            } else {
                threads_per_stream = threads_per_stream > 1 ? threads_per_stream - 1 : 1;
            }
        }

        if (threads_per_stream == 1) {
            const int stream_loops = streams / num_cores;
            const int remain_streams = streams - stream_loops * num_cores;
            if (num_big_cores_phys >= remain_streams) {
                config.set_property(Config::big_core_streams.name(), remain_streams + num_big_cores * stream_loops);
                config.set_property(Config::small_core_streams.name(), num_small_cores_phys * stream_loops);
            } else if (num_big_cores_phys + num_small_cores_phys >= remain_streams) {
                config.set_property(Config::big_core_streams.name(), num_big_cores_phys + num_big_cores * stream_loops);
                config.set_property(Config::small_core_streams.name(),
                                    remain_streams - num_big_cores_phys + num_small_cores_phys * stream_loops);
            } else {
                config.set_property(Config::big_core_streams.name(),
                                    remain_streams - num_small_cores_phys + num_big_cores * stream_loops);
                config.set_property(Config::small_core_streams.name(), num_small_cores_phys * (stream_loops + 1));
            }
        }

        config.set_property(Config::threads_per_stream_big.name(), threads_per_stream);
        config.set_property(Config::threads_per_stream_small.name(), threads_per_stream);
    }
}

IStreamsExecutor::Config IStreamsExecutor::Config::make_default_multi_threaded(const IStreamsExecutor::Config& initial,
                                                                               const bool fp_intesive) {
    const auto envThreads = parallel_get_env_threads();
    const auto& numaNodes = get_available_numa_nodes();
    const int numaNodesNum = static_cast<int>(numaNodes.size());
    auto streamExecutorConfig = initial;
    auto _streams = streamExecutorConfig.get_property(Config::streams.name()).as<int32_t>();
    auto _threads = streamExecutorConfig.get_property(Config::threads.name()).as<int32_t>();
    auto _big_core_streams = streamExecutorConfig.get_property(Config::big_core_streams.name()).as<int32_t>();
    auto _small_core_streams = streamExecutorConfig.get_property(Config::small_core_streams.name()).as<int32_t>();
    auto _threadsPerStream = streamExecutorConfig.get_property(Config::threads_per_stream.name()).as<int32_t>();
    auto _threads_per_stream_small =
        streamExecutorConfig.get_property(Config::threads_per_stream_small.name()).as<int32_t>();
    auto _threads_per_stream_big =
        streamExecutorConfig.get_property(Config::threads_per_stream_big.name()).as<int32_t>();
    auto _threadBindingType =
        streamExecutorConfig.get_property(Config::thread_binding_type.name()).as<ThreadBindingType>();
    const bool bLatencyCase = _streams <= numaNodesNum;

    // by default, do not use the hyper-threading (to minimize threads synch overheads)
    int num_cores_default = get_number_of_cpu_cores();
#if (OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO)
    // additional latency-case logic for hybrid processors:
    if (ThreadBindingType::HYBRID_AWARE == _threadBindingType) {
        const auto core_types = custom::info::core_types();
        const auto num_little_cores =
            custom::info::default_concurrency(custom::task_arena::constraints{}.set_core_type(core_types.front()));
        const auto num_big_cores_phys = get_number_of_cpu_cores(true);
        const int int8_threshold = 4;  // ~relative efficiency of the VNNI-intensive code for Big vs Little cores;
        const int fp32_threshold = 2;  // ~relative efficiency of the AVX2 fp32 code for Big vs Little cores;
        // by default the latency case uses (faster) Big cores only, depending on the compute ratio
        const bool bLatencyCaseBigOnly =
            num_big_cores_phys > (num_little_cores / (fp_intesive ? fp32_threshold : int8_threshold));
        // selecting the preferred core type
        streamExecutorConfig.set_property(Config::thread_preferred_core_type.name(),
                                          bLatencyCase
                                              ? (bLatencyCaseBigOnly ? IStreamsExecutor::Config::PreferredCoreType::BIG
                                                                     : IStreamsExecutor::Config::PreferredCoreType::ANY)
                                              : IStreamsExecutor::Config::PreferredCoreType::ROUND_ROBIN);
        // additionally selecting the #cores to use in the "Big-only" case
        if (bLatencyCaseBigOnly) {
            const int hyper_threading_threshold =
                2;  // min #cores, for which the hyper-threading becomes useful for the latency case
            const auto num_big_cores =
                custom::info::default_concurrency(custom::task_arena::constraints{}.set_core_type(core_types.back()));
            num_cores_default = (num_big_cores_phys <= hyper_threading_threshold) ? num_big_cores : num_big_cores_phys;
        }
        // if nstreams or nthreads are set, need to calculate the Hybrid aware parameters here
        if (!bLatencyCase && (_big_core_streams == 0 || _threads)) {
            update_hybrid_custom_threads(streamExecutorConfig);
        }
        OPENVINO_DEBUG << "[ p_e_core_info ] streams (threads): " << _streams << "("
                       << _threads_per_stream_big * _big_core_streams + _threads_per_stream_small * _small_core_streams
                       << ") -- PCore: " << _big_core_streams << "(" << _threads_per_stream_big
                       << ")  ECore: " << _small_core_streams << "(" << _threads_per_stream_small << ")";
    }
#endif
    const auto hwCores = !bLatencyCase && numaNodesNum == 1
                             // throughput case on a single-NUMA node machine uses all available cores
                             ? (streamExecutorConfig.get_property(Config::enable_hyper_thread.name()).as<bool>()
                                    ? parallel_get_max_threads()
                                    : num_cores_default)
                             // in the rest of cases:
                             //    multi-node machine
                             //    or
                             //    latency case, single-node yet hybrid case that uses
                             //      all core types
                             //      or
                             //      big-cores only, but the #cores is "enough" (pls see the logic above)
                             // it is usually beneficial not to use the hyper-threading (which is default)
                             : num_cores_default;
    const auto threads = _threads ? _threads : (envThreads ? envThreads : hwCores);
    _threadsPerStream = _streams ? std::max(1, threads / _streams) : threads;
    _threads = (!bLatencyCase && ThreadBindingType::HYBRID_AWARE == _threadBindingType)
                   ? _big_core_streams * _threads_per_stream_big + _small_core_streams * _threads_per_stream_small
                   : _threadsPerStream * _streams;
    return streamExecutorConfig;
}

}  // namespace threading
}  // namespace ov
