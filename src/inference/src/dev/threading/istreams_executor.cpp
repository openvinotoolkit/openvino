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
#include "openvino/runtime/threading/cpu_streams_info.hpp"
#include "openvino/util/log.hpp"
#include "parallel_custom_arena.hpp"

namespace ov {
namespace threading {

IStreamsExecutor::~IStreamsExecutor() {}

void IStreamsExecutor::Config::set_property(const std::string& key, const ov::Any& value) {
    set_property({{key, value}});
}

void IStreamsExecutor::Config::set_property(const ov::AnyMap& property) {
    for (const auto& it : property) {
        const auto& key = it.first;
        const auto value = it.second;
        if (key == ov::num_streams) {
            auto streams = value.as<ov::streams::Num>();
            if (streams.num >= 0) {
                _streams = streams.num;
                _streams_changed = true;
            } else if (streams.num < ov::streams::NUMA) {
                OPENVINO_THROW("Wrong value for property key ",
                               ov::num_streams.name(),
                               ". Expected non negative numbers (#streams) or ",
                               "ov::streams::NUMA|ov::streams::AUTO, Got: ",
                               streams);
            }
        } else if (key == ov::inference_num_threads) {
            int val_i;
            try {
                val_i = value.as<int>();
            } catch (const std::exception&) {
                OPENVINO_THROW("Wrong value for property key ",
                               ov::inference_num_threads.name(),
                               ". Expected only positive numbers (#threads)");
            }
            if (val_i < 0) {
                OPENVINO_THROW("Wrong value for property key ",
                               ov::inference_num_threads.name(),
                               ". Expected only positive numbers (#threads)");
            }
            _threads = val_i;
        } else if (key == ov::internal::threads_per_stream) {
            _threadsPerStream = static_cast<int>(value.as<size_t>());
        } else {
            OPENVINO_THROW("Wrong value for property key ", key);
        }
    }
}

ov::Any IStreamsExecutor::Config::get_property(const std::string& key) const {
    if (key == ov::supported_properties) {
        std::vector<std::string> properties{
            ov::num_streams.name(),
            ov::inference_num_threads.name(),
            ov::internal::threads_per_stream.name(),
            ov::affinity.name(),
        };
        return properties;
    } else if (key == ov::affinity) {
        switch (_threadBindingType) {
        case IStreamsExecutor::ThreadBindingType::NONE:
            return ov::Affinity::NONE;
        case IStreamsExecutor::ThreadBindingType::CORES:
            return ov::Affinity::CORE;
        case IStreamsExecutor::ThreadBindingType::NUMA:
            return ov::Affinity::NUMA;
        case IStreamsExecutor::ThreadBindingType::HYBRID_AWARE:
            return ov::Affinity::HYBRID_AWARE;
        }
    } else if (key == ov::num_streams) {
        return decltype(ov::num_streams)::value_type{_streams};
    } else if (key == ov::inference_num_threads) {
        return decltype(ov::inference_num_threads)::value_type{_threads};
    } else if (key == ov::internal::threads_per_stream) {
        return decltype(ov::internal::threads_per_stream)::value_type{_threadsPerStream};
    } else {
        OPENVINO_THROW("Wrong value for property key ", key);
    }
    return {};
}

IStreamsExecutor::Config IStreamsExecutor::Config::make_default_multi_threaded(
    const IStreamsExecutor::Config& initial) {
    const auto proc_type_table = get_proc_type_table();
    auto streamConfig = initial;

    if (proc_type_table.empty()) {
        return streamConfig;
    }

    const auto numa_nodes = proc_type_table.size() > 1 ? proc_type_table.size() - 1 : proc_type_table.size();
    const bool latency_case = static_cast<size_t>(streamConfig._streams) <= numa_nodes;

    // by default, do not use the hyper-threading (to minimize threads synch overheads)
    int num_cores = !latency_case && numa_nodes == 1
                        ? proc_type_table[0][ALL_PROC]
                        : proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][EFFICIENT_CORE_PROC];

    // additional latency-case logic for hybrid processors:
    if (proc_type_table[0][EFFICIENT_CORE_PROC] > 0 && proc_type_table[0][MAIN_CORE_PROC] > 0) {
        if (streamConfig._threadPreferredCoreType == IStreamsExecutor::Config::ANY) {
            // by default the latency case uses (faster) Big cores only, depending on the compute ratio
            const bool big_only = proc_type_table[0][MAIN_CORE_PROC] > (proc_type_table[0][EFFICIENT_CORE_PROC] / 2);
            // selecting the preferred core type
            if (big_only) {
                streamConfig._threadPreferredCoreType = IStreamsExecutor::Config::PreferredCoreType::BIG;
                const int hyper_threading_threshold =
                    2;  // min #cores, for which the hyper-threading becomes useful for the latency case
                // additionally selecting the #cores to use in the "Big-only" case
                num_cores = (proc_type_table[0][MAIN_CORE_PROC] <= hyper_threading_threshold)
                                ? proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][HYPER_THREADING_PROC]
                                : proc_type_table[0][MAIN_CORE_PROC];
            }
        } else if (streamConfig._threadPreferredCoreType == IStreamsExecutor::Config::BIG) {
            num_cores = proc_type_table[0][MAIN_CORE_PROC];
        } else if (streamConfig._threadPreferredCoreType == IStreamsExecutor::Config::LITTLE) {
            num_cores = proc_type_table[0][EFFICIENT_CORE_PROC];
        }
    }

    const auto threads = streamConfig._threads ? streamConfig._threads : num_cores;
    int threads_per_stream = streamConfig._streams ? std::max(1, threads / streamConfig._streams) : threads;
    if (proc_type_table[0][EFFICIENT_CORE_PROC] > 0 && proc_type_table[0][MAIN_CORE_PROC] > 0 &&
        streamConfig._threadPreferredCoreType == IStreamsExecutor::Config::ANY) {
        if (streamConfig._streams > 1) {
            threads_per_stream =
                std::min(std::min(proc_type_table[0][MAIN_CORE_PROC], proc_type_table[0][EFFICIENT_CORE_PROC]),
                         threads_per_stream);
            while (1) {
                int streams_num = proc_type_table[0][MAIN_CORE_PROC] / threads_per_stream +
                                  proc_type_table[0][HYPER_THREADING_PROC] / threads_per_stream +
                                  proc_type_table[0][EFFICIENT_CORE_PROC] / threads_per_stream;
                if (streams_num >= streamConfig._streams) {
                    break;
                } else {
                    if (threads_per_stream > 1) {
                        threads_per_stream--;
                    }
                }
            }
        }
    }
    streamConfig._threadsPerStream = threads_per_stream;
    streamConfig._threads = streamConfig._threadsPerStream * streamConfig._streams;
    return streamConfig;
}

IStreamsExecutor::Config IStreamsExecutor::Config::reserve_cpu_threads(const IStreamsExecutor::Config& initial) {
    auto config = initial;
    int status = config._name.find("StreamsExecutor") != std::string::npos ? NOT_USED : CPU_USED;

    if (config._streams_info_table.size() == 0 || (status == CPU_USED && !config._cpu_reservation)) {
        return config;
    }

    reserve_available_cpus(config._streams_info_table, config._stream_processor_ids, status);

    config._streams = 0;
    config._threads = 0;
    for (size_t i = 0; i < config._streams_info_table.size(); i++) {
        if (config._streams_info_table[i][NUMBER_OF_STREAMS] > 0) {
            config._streams += config._streams_info_table[i][NUMBER_OF_STREAMS];
            config._threads +=
                config._streams_info_table[i][NUMBER_OF_STREAMS] * config._streams_info_table[i][THREADS_PER_STREAM];
        }
    }
    OPENVINO_DEBUG << "[ threading ] " << config._name << " reserve_cpu_threads " << config._streams << "("
                   << config._threads << ")";

    return config;
}

void IStreamsExecutor::Config::update_executor_config(int stream_nums,
                                                      int threads_per_stream,
                                                      IStreamsExecutor::Config::PreferredCoreType core_type,
                                                      bool cpu_pinning) {
    const auto proc_type_table = ov::get_proc_type_table();

    if (proc_type_table.empty()) {
        return;
    }

    if (proc_type_table.size() > 1) {
        core_type = ov::threading::IStreamsExecutor::Config::ANY;
    }

    // IStreamsExecutor::Config config = initial;
    const auto total_num_cores = proc_type_table[0][ALL_PROC];
    const auto total_num_big_cores = proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][HYPER_THREADING_PROC];
    const auto total_num_little_cores = proc_type_table[0][EFFICIENT_CORE_PROC];

    int num_cores = total_num_cores;
    if (core_type == ov::threading::IStreamsExecutor::Config::BIG) {
        num_cores = total_num_big_cores;
    } else if (core_type == ov::threading::IStreamsExecutor::Config::LITTLE) {
        num_cores = total_num_little_cores;
    }

    int streams = std::min(stream_nums, num_cores);

    if (streams == 0) {
        return;
    }

    _streams = streams;
    _threadPreferredCoreType = core_type;
    _threadsPerStream = threads_per_stream;

    // create stream_info_table based on core type
    std::vector<int> stream_info(ov::CPU_STREAMS_TABLE_SIZE, 0);
    stream_info[ov::THREADS_PER_STREAM] = _threadsPerStream;
    stream_info[ov::STREAM_NUMA_NODE_ID] = 0;
    stream_info[ov::STREAM_SOCKET_ID] = 0;
    if (core_type == ov::threading::IStreamsExecutor::Config::BIG) {
        if (proc_type_table[0][ov::MAIN_CORE_PROC] < _streams) {
            if (proc_type_table[0][ov::MAIN_CORE_PROC] > 0) {
                stream_info[ov::NUMBER_OF_STREAMS] = proc_type_table[0][ov::MAIN_CORE_PROC];
                stream_info[ov::PROC_TYPE] = ov::MAIN_CORE_PROC;
                _streams_info_table.push_back(stream_info);
            }
            if (proc_type_table[0][ov::HYPER_THREADING_PROC] > 0) {
                stream_info[ov::NUMBER_OF_STREAMS] = proc_type_table[0][ov::HYPER_THREADING_PROC];
                stream_info[ov::PROC_TYPE] = ov::HYPER_THREADING_PROC;
                _streams_info_table.push_back(stream_info);
            }
        } else {
            stream_info[ov::PROC_TYPE] = ov::MAIN_CORE_PROC;
            stream_info[ov::NUMBER_OF_STREAMS] = _streams;
            _streams_info_table.push_back(stream_info);
        }
    } else if (core_type == ov::threading::IStreamsExecutor::Config::LITTLE) {
        stream_info[ov::PROC_TYPE] = ov::EFFICIENT_CORE_PROC;
        stream_info[ov::NUMBER_OF_STREAMS] = _streams;
        _streams_info_table.push_back(stream_info);
    } else {
        int total_streams = 0;
        if (proc_type_table.size() == 1) {
            for (int i = ov::MAIN_CORE_PROC; i <= ov::HYPER_THREADING_PROC; i++) {
                if (proc_type_table[0][i] > 0) {
                    stream_info[ov::NUMBER_OF_STREAMS] =
                        (total_streams + proc_type_table[0][i] > _streams ? _streams - total_streams
                                                                          : proc_type_table[0][i]);
                    stream_info[ov::PROC_TYPE] = i;
                    stream_info[ov::STREAM_NUMA_NODE_ID] = proc_type_table[0][PROC_NUMA_NODE_ID];
                    stream_info[ov::STREAM_SOCKET_ID] = proc_type_table[0][PROC_SOCKET_ID];
                    _streams_info_table.push_back(stream_info);
                    total_streams += stream_info[ov::NUMBER_OF_STREAMS];
                }
                if (total_streams >= _streams)
                    break;
            }
        } else {
            for (size_t i = 1; i < proc_type_table.size(); i++) {
                for (int j = ov::MAIN_CORE_PROC; j < ov::HYPER_THREADING_PROC; j++) {
                    if (proc_type_table[i][j] > 0) {
                        stream_info[ov::NUMBER_OF_STREAMS] =
                            (total_streams + proc_type_table[i][j] > _streams ? _streams - total_streams
                                                                              : proc_type_table[i][j]);
                        stream_info[ov::PROC_TYPE] = j;
                        stream_info[ov::STREAM_NUMA_NODE_ID] = proc_type_table[i][PROC_NUMA_NODE_ID];
                        stream_info[ov::STREAM_SOCKET_ID] = proc_type_table[i][PROC_SOCKET_ID];
                        _streams_info_table.push_back(stream_info);
                        total_streams += stream_info[ov::NUMBER_OF_STREAMS];
                    }
                    if (total_streams >= _streams)
                        break;
                }
                if (total_streams >= _streams)
                    break;
            }
        }
    }

    if (cpu_pinning) {
        _cpu_reservation = cpu_pinning;
        auto new_config = reserve_cpu_threads(*this);
        _stream_processor_ids = new_config._stream_processor_ids;
        _streams = new_config._streams;
        _threads = new_config._threads;
    }
}

void IStreamsExecutor::Config::set_config_zero_stream() {
    std::vector<std::vector<int>> proc_type_table = get_proc_type_table();
    int core_type = MAIN_CORE_PROC;
    int numa_id = 0;
    int socket_id = 0;

    if (proc_type_table.size() > 0) {
        core_type = proc_type_table[0][MAIN_CORE_PROC] > 0
                        ? MAIN_CORE_PROC
                        : (proc_type_table[0][EFFICIENT_CORE_PROC] > 0 ? EFFICIENT_CORE_PROC : HYPER_THREADING_PROC);
        numa_id = std::max(0, proc_type_table[0][PROC_NUMA_NODE_ID]);
        socket_id = std::max(0, proc_type_table[0][PROC_SOCKET_ID]);
    }
    _streams_info_table.push_back({1, core_type, 1, numa_id, socket_id});
    _cpu_reservation = false;
}

}  // namespace threading
}  // namespace ov
