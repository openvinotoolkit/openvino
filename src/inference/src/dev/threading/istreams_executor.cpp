// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/threading/istreams_executor.hpp"

#include <algorithm>
#include <future>
#include <string>
#include <thread>
#include <vector>

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
        const auto& value = it.second;
        if (key == ov::num_streams) {
            auto streams = value.as<ov::streams::Num>();
            if (streams == ov::streams::NUMA) {
                _streams = get_num_numa_nodes();
            } else if (streams == ov::streams::AUTO) {
                // bare minimum of streams (that evenly divides available number of cores)
                _streams = get_default_num_streams();
            } else if (streams.num >= 0) {
                _streams = streams.num;
            } else {
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
            _threads_per_stream = static_cast<int>(value.as<size_t>());
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
        };
        return properties;
    } else if (key == ov::num_streams) {
        return decltype(ov::num_streams)::value_type{_streams};
    } else if (key == ov::inference_num_threads) {
        return decltype(ov::inference_num_threads)::value_type{_threads};
    } else if (key == ov::internal::threads_per_stream) {
        return decltype(ov::internal::threads_per_stream)::value_type{_threads_per_stream};
    } else {
        OPENVINO_THROW("Wrong value for property key ", key);
    }
    return {};
}

int IStreamsExecutor::Config::get_default_num_streams() {
    // bare minimum of streams (that evenly divides available number of core)
    const auto proc_type_table = get_proc_type_table();
    if (proc_type_table.empty()) {
        return 1;
    }
    const auto num_cores = proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][EFFICIENT_CORE_PROC];
    if (0 == num_cores % 4)
        return std::max(4, num_cores / 4);
    else if (0 == num_cores % 5)
        return std::max(5, num_cores / 5);
    else if (0 == num_cores % 3)
        return std::max(3, num_cores / 3);
    else  // if user disables some cores say in BIOS, so we got weird #cores which is not easy to divide
        return 1;
}

IStreamsExecutor::Config IStreamsExecutor::Config::make_default_multi_threaded(
    const IStreamsExecutor::Config& initial) {
    const auto proc_type_table = get_proc_type_table();
    auto streamConfig = initial;

    if (proc_type_table.empty()) {
        return streamConfig;
    }

    int num_cores = proc_type_table[0][ALL_PROC];

    if (proc_type_table[0][EFFICIENT_CORE_PROC] > 0 && proc_type_table[0][MAIN_CORE_PROC] > 0) {
        if (streamConfig._thread_preferred_core_type == ov::hint::SchedulingCoreType::ANY_CORE) {
            num_cores = proc_type_table[0][ALL_PROC];
        } else if (streamConfig._thread_preferred_core_type == ov::hint::SchedulingCoreType::PCORE_ONLY) {
            num_cores = proc_type_table[0][MAIN_CORE_PROC];
        } else if (streamConfig._thread_preferred_core_type == ov::hint::SchedulingCoreType::ECORE_ONLY) {
            num_cores = proc_type_table[0][EFFICIENT_CORE_PROC];
        }
    }

    const auto threads = streamConfig._threads ? streamConfig._threads : num_cores;
    int threads_per_stream = streamConfig._streams ? std::max(1, threads / streamConfig._streams) : threads;
    if (proc_type_table[0][EFFICIENT_CORE_PROC] > 0 && proc_type_table[0][MAIN_CORE_PROC] > 0 &&
        streamConfig._thread_preferred_core_type == ov::hint::SchedulingCoreType::ANY_CORE) {
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
    streamConfig._threads_per_stream = threads_per_stream;
    streamConfig._threads = streamConfig._threads_per_stream * streamConfig._streams;
    streamConfig.update_executor_config();
    return streamConfig;
}

void IStreamsExecutor::Config::update_executor_config() {
    const auto proc_type_table = get_proc_type_table();
    bool streams_info_available = false;

    if (proc_type_table.empty()) {
        return;
    }

    if (_cpu_reservation && !_cpu_pinning) {
        _cpu_pinning = true;
    }

    if (!_streams_info_table.empty()) {
        streams_info_available = true;
        std::vector<int> threads_proc_type(HYPER_THREADING_PROC + 1, 0);
        for (size_t i = 0; i < _streams_info_table.size(); i++) {
            if (_streams_info_table[i][NUMBER_OF_STREAMS] > 0) {
                threads_proc_type[_streams_info_table[i][PROC_TYPE]] +=
                    _streams_info_table[i][THREADS_PER_STREAM] * _streams_info_table[i][NUMBER_OF_STREAMS];
            }
        }
        for (size_t i = ALL_PROC; i < threads_proc_type.size(); i++) {
            if (threads_proc_type[i] > proc_type_table[0][i]) {
                streams_info_available = false;
                break;
            }
        }
    }

    if (!streams_info_available) {
        _streams_info_table.clear();

        const auto total_num_cores = proc_type_table[0][ALL_PROC];
        const auto total_num_big_cores = proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][HYPER_THREADING_PROC];
        const auto total_num_little_cores = proc_type_table[0][EFFICIENT_CORE_PROC];

        if ((total_num_little_cores == 0 && _thread_preferred_core_type == ov::hint::SchedulingCoreType::ECORE_ONLY) ||
            (total_num_big_cores == 0 && _thread_preferred_core_type == ov::hint::SchedulingCoreType::PCORE_ONLY) ||
            (proc_type_table.size() > 1 && _thread_preferred_core_type == ov::hint::SchedulingCoreType::PCORE_ONLY)) {
            _thread_preferred_core_type = ov::hint::SchedulingCoreType::ANY_CORE;
        }

        int num_cores = total_num_cores;
        if (_thread_preferred_core_type == ov::hint::SchedulingCoreType::PCORE_ONLY) {
            num_cores = total_num_big_cores;
        } else if (_thread_preferred_core_type == ov::hint::SchedulingCoreType::ECORE_ONLY) {
            num_cores = total_num_little_cores;
        }

        _streams = _streams > 0 ? std::min(_streams, num_cores) : _streams;
        if (_streams == 0) {
            set_config_zero_stream();
            return;
        }

        _threads_per_stream =
            _threads_per_stream > 0 ? std::min(num_cores, _streams * _threads_per_stream) / _streams : 0;
        if (_threads_per_stream == 0) {
            return;
        }

        // create stream_info_table based on core type
        std::vector<int> stream_info(CPU_STREAMS_TABLE_SIZE, 0);
        stream_info[THREADS_PER_STREAM] = _threads_per_stream;
        stream_info[STREAM_NUMA_NODE_ID] = 0;
        stream_info[STREAM_SOCKET_ID] = 0;
        int cur_threads = _streams * _threads_per_stream;
        if (_thread_preferred_core_type == ov::hint::SchedulingCoreType::ECORE_ONLY) {
            stream_info[PROC_TYPE] = EFFICIENT_CORE_PROC;
            stream_info[NUMBER_OF_STREAMS] = _streams;
            _streams_info_table.push_back(stream_info);
        } else {
            int start = proc_type_table.size() > 1 ? 1 : 0;
            std::vector<int> core_types;
            // Using cores crossed sockets or hyper threads when streams = 1
            if (_streams == 1 && _threads_per_stream > proc_type_table[start][ov::MAIN_CORE_PROC]) {
                stream_info[NUMBER_OF_STREAMS] = _streams;
                stream_info[PROC_TYPE] = ALL_PROC;
                stream_info[STREAM_NUMA_NODE_ID] = proc_type_table.size() > 1 ? -1 : 0;
                stream_info[STREAM_SOCKET_ID] = proc_type_table.size() > 1 ? -1 : 0;
                _streams_info_table.push_back(stream_info);
                stream_info[NUMBER_OF_STREAMS] = 0;
            }
            if (_thread_preferred_core_type == ov::hint::SchedulingCoreType::PCORE_ONLY &&
                proc_type_table[0][EFFICIENT_CORE_PROC] > 0) {
                core_types = {MAIN_CORE_PROC, HYPER_THREADING_PROC};
            } else {
                core_types = {MAIN_CORE_PROC, EFFICIENT_CORE_PROC, HYPER_THREADING_PROC};
            }
            for (int j : core_types) {
                for (size_t i = start; i < proc_type_table.size(); i++) {
                    if (proc_type_table[i][j] > 0 && cur_threads > 0) {
                        if (_threads_per_stream > proc_type_table[i][j]) {
                            stream_info[THREADS_PER_STREAM] = std::min(proc_type_table[i][j], cur_threads);
                            cur_threads -= stream_info[THREADS_PER_STREAM];
                        } else {
                            stream_info[NUMBER_OF_STREAMS] =
                                std::min(proc_type_table[i][j], cur_threads) / _threads_per_stream;
                            cur_threads -= stream_info[NUMBER_OF_STREAMS] * _threads_per_stream;
                        }
                        stream_info[PROC_TYPE] = j;
                        stream_info[STREAM_NUMA_NODE_ID] = proc_type_table[i][PROC_NUMA_NODE_ID];
                        stream_info[STREAM_SOCKET_ID] = proc_type_table[i][PROC_SOCKET_ID];
                        _streams_info_table.push_back(stream_info);
                    }
                }
            }
        }
    }

    if (_cpu_pinning) {
        reserve_available_cpus(_streams_info_table, _stream_processor_ids, _cpu_reservation ? CPU_USED : NOT_USED);
    }

    // Recaculate _streams, _threads and _threads_per_stream by _streams_info_table
    int num_streams = 0;
    _threads = 0;
    _sub_streams = 0;
    for (size_t i = 0; i < _streams_info_table.size(); i++) {
        if (_streams_info_table[i][NUMBER_OF_STREAMS] > 0) {
            num_streams += _streams_info_table[i][NUMBER_OF_STREAMS];
            _threads += _streams_info_table[i][NUMBER_OF_STREAMS] * _streams_info_table[i][THREADS_PER_STREAM];
        } else if (_streams_info_table[i][NUMBER_OF_STREAMS] == -1) {
            _sub_streams += 1;
        }
    }
    _threads_per_stream = _streams_info_table[0][THREADS_PER_STREAM];
    _streams = _streams > 0 ? num_streams : _streams;

#ifdef ENABLE_OPENVINO_DEBUG
    OPENVINO_DEBUG("[ threading ] proc_type_table:");
    for (size_t i = 0; i < proc_type_table.size(); i++) {
        OPENVINO_DEBUG(proc_type_table[i][ALL_PROC],
                       proc_type_table[i][MAIN_CORE_PROC],
                       " ",
                       proc_type_table[i][EFFICIENT_CORE_PROC],
                       " ",
                       proc_type_table[i][HYPER_THREADING_PROC],
                       " ",
                       proc_type_table[i][PROC_NUMA_NODE_ID],
                       " ",
                       proc_type_table[i][PROC_SOCKET_ID]);
    }

    OPENVINO_DEBUG("[ threading ] streams_info_table:");
    for (size_t i = 0; i < _streams_info_table.size(); i++) {
        OPENVINO_DEBUG(_streams_info_table[i][NUMBER_OF_STREAMS],
                       " ",
                       _streams_info_table[i][PROC_TYPE],
                       " ",
                       _streams_info_table[i][THREADS_PER_STREAM],
                       " ",
                       _streams_info_table[i][STREAM_NUMA_NODE_ID],
                       " ",
                       _streams_info_table[i][STREAM_SOCKET_ID]);
    }
    OPENVINO_DEBUG("[ threading ] )", _name, ": ", _streams, "(", _threads, ")");
#endif
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
    _cpu_pinning = false;
}

}  // namespace threading
}  // namespace ov
