// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/threading/cpu_streams_executor_internal.hpp"

#include <algorithm>
#include <vector>

#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/threading/cpu_streams_info.hpp"

namespace ov {
namespace threading {

void get_cur_stream_info(const int stream_id,
                         const bool cpu_pinning,
                         const std::vector<std::vector<int>> proc_type_table,
                         const std::vector<std::vector<int>> streams_info_table,
                         StreamCreateType& stream_type,
                         int& concurrency,
                         int& core_type,
                         int& numa_node_id,
                         int& max_threads_per_core) {
    int stream_total = 0;
    size_t stream_info_id = 0;
    bool pinning = cpu_pinning;
    bool ecore_used = false;
    for (size_t i = 0; i < streams_info_table.size(); i++) {
        stream_total += std::abs(streams_info_table[i][NUMBER_OF_STREAMS]);
        if (stream_id < stream_total) {
            stream_info_id = i;
            break;
        }
    }
    concurrency = streams_info_table[stream_info_id][THREADS_PER_STREAM];
    core_type = streams_info_table[stream_info_id][PROC_TYPE];
    numa_node_id = streams_info_table[stream_info_id][STREAM_NUMA_NODE_ID];
    max_threads_per_core = 1;
    if (core_type == ALL_PROC) {
        for (size_t i = stream_info_id + 1; i < streams_info_table.size(); i++) {
            if (streams_info_table[i][NUMBER_OF_STREAMS] == 0) {
                if (streams_info_table[i][PROC_TYPE] == EFFICIENT_CORE_PROC) {
                    ecore_used = true;
                } else if (streams_info_table[i][PROC_TYPE] == HYPER_THREADING_PROC) {
                    max_threads_per_core = 2;
                }
            } else {
                break;
            }
        }
    } else if (core_type == HYPER_THREADING_PROC) {
        max_threads_per_core = 2;
    }

#if defined(__APPLE__)
    pinning = false;
#elif defined(_WIN32)
    if (proc_type_table.size() > 1) {
        pinning = false;
    }
#endif
    if (pinning) {
        stream_type = STREAM_WITH_OBSERVE;
    } else {
        stream_type = STREAM_WITHOUT_PARAM;
        // Pcore only or Ecore only with no cpu binding in hybrid cores machine
        if (proc_type_table[0][EFFICIENT_CORE_PROC] > 0 && core_type != ALL_PROC) {
            stream_type = STREAM_WITH_CORE_TYPE;
        } else if (proc_type_table[0][EFFICIENT_CORE_PROC] > 0 && core_type == ALL_PROC &&
                   !ecore_used) {  // Latency mode and enable hyper threading in hybrid cores machine
            stream_type = STREAM_WITH_CORE_TYPE;
            core_type = MAIN_CORE_PROC;
        } else if (proc_type_table.size() > 1 && numa_node_id >= 0) {
            stream_type = STREAM_WITH_NUMA_ID;
        }
    }
}

void reserve_cpu_by_streams_info(const std::vector<std::vector<int>> _streams_info_table,
                                 const int _numa_nodes,
                                 std::vector<std::vector<int>>& _cpu_mapping_table,
                                 std::vector<std::vector<int>>& _proc_type_table,
                                 std::vector<std::vector<int>>& _stream_processors,
                                 const int _cpu_status) {
    std::vector<std::vector<int>> streams_table;
    std::vector<std::vector<std::pair<std::string, int>>> stream_conditions;
    std::vector<std::vector<int>> threads_status;  // used to count the number of threads
    std::vector<int> stream_pos;
    std::vector<int> stream_num;
    int num_streams = 0;
    int num_conditions = 0;
    int condition_idx = 0;
    bool last_all_proc = false;
    bool sub_stream_enable = false;

    for (size_t i = 0; i < _streams_info_table.size(); i++) {
        if (i > 0 && _streams_info_table[i][NUMBER_OF_STREAMS] < 0 &&
            _streams_info_table[i - 1][NUMBER_OF_STREAMS] > 0) {
            stream_pos.clear();
            num_streams = 0;
            sub_stream_enable = true;
        }
        if (_streams_info_table[i][NUMBER_OF_STREAMS] != 0) {
            stream_pos.push_back(num_streams);
        }
        num_streams += std::abs(_streams_info_table[i][NUMBER_OF_STREAMS]);
    }
    num_conditions = static_cast<int>(stream_pos.size());
    _stream_processors.assign(num_streams, std::vector<int>());
    stream_conditions.assign(num_conditions, std::vector<std::pair<std::string, int>>());
    threads_status.assign(num_conditions, std::vector<int>());
    stream_num.assign(num_conditions, 0);

    for (size_t i = 0; i < _streams_info_table.size(); i++) {
        std::string proc_type = "";
        std::string numa_node = "";
        std::string socket = "";
        if ((_streams_info_table[i][NUMBER_OF_STREAMS] > 0 && !sub_stream_enable) ||
            (_streams_info_table[i][NUMBER_OF_STREAMS] < 0 && sub_stream_enable)) {
            streams_table.push_back(_streams_info_table[i]);
            if (_streams_info_table[i][NUMBER_OF_STREAMS] < 0) {
                streams_table[streams_table.size() - 1][NUMBER_OF_STREAMS] =
                    std::abs(_streams_info_table[i][NUMBER_OF_STREAMS]);
            }
        } else if (_streams_info_table[i][NUMBER_OF_STREAMS] != 0) {
            continue;
        }
        if (last_all_proc && _streams_info_table[i][NUMBER_OF_STREAMS] != 0) {
            last_all_proc = false;
            condition_idx++;
        }
        if (_streams_info_table[i][PROC_TYPE] > ALL_PROC) {
            proc_type = std::to_string(_streams_info_table[i][PROC_TYPE]);
        } else {
            last_all_proc = true;
        }
        if (_streams_info_table[i][STREAM_NUMA_NODE_ID] >= 0) {
            numa_node = std::to_string(_streams_info_table[i][STREAM_NUMA_NODE_ID]);
        }
        if (_streams_info_table[i][STREAM_SOCKET_ID] >= 0) {
            socket = std::to_string(_streams_info_table[i][STREAM_SOCKET_ID]);
        }
        if (proc_type != "") {
            stream_conditions[condition_idx].push_back(
                std::make_pair(proc_type + numa_node + socket, _streams_info_table[i][THREADS_PER_STREAM]));
            threads_status[condition_idx].push_back(0);
        }
        if (_streams_info_table[i][PROC_TYPE] > ALL_PROC && _streams_info_table[i][NUMBER_OF_STREAMS] != 0) {
            condition_idx++;
        }
    }

    for (size_t i = 0; i < _cpu_mapping_table.size(); i++) {
        if (_cpu_mapping_table[i][CPU_MAP_USED_FLAG] == NOT_USED) {
            std::string cpu_string = std::to_string(_cpu_mapping_table[i][CPU_MAP_CORE_TYPE]) +
                                     std::to_string(_cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID]) +
                                     std::to_string(_cpu_mapping_table[i][CPU_MAP_SOCKET_ID]);
            for (size_t j = 0; j < stream_conditions.size(); j++) {
                auto iter = std::find_if(stream_conditions[j].begin(),
                                         stream_conditions[j].end(),
                                         [&](std::pair<std::string, int> item) {
                                             return cpu_string == item.first;
                                         });
                if (iter != stream_conditions[j].end()) {
                    // process the situation of proc_type = ALL_PROC
                    if (stream_conditions[j].size() > 1) {
                        size_t idx = iter - stream_conditions[j].begin();
                        threads_status[j][idx]++;
                        if (threads_status[j][idx] >= stream_conditions[j][idx].second) {
                            stream_conditions[j][idx] = std::make_pair("", 0);
                        }
                    }
                    _stream_processors[stream_pos[j]].push_back(_cpu_mapping_table[i][CPU_MAP_PROCESSOR_ID]);
                    _cpu_mapping_table[i][CPU_MAP_USED_FLAG] = _cpu_status;
                    if (static_cast<int>(_stream_processors[stream_pos[j]].size()) ==
                        streams_table[j][THREADS_PER_STREAM]) {
                        stream_pos[j]++;
                        stream_num[j]++;
                    }
                    if (stream_num[j] >= streams_table[j][NUMBER_OF_STREAMS]) {
                        stream_conditions[j].clear();
                    }
                    break;
                }
            }
        }
    }

    if (_cpu_status > NOT_USED) {
        update_proc_type_table(_cpu_mapping_table, _numa_nodes, _proc_type_table);
    }
}

void update_proc_type_table(const std::vector<std::vector<int>> _cpu_mapping_table,
                            const int _numa_nodes,
                            std::vector<std::vector<int>>& _proc_type_table) {
    std::map<int, int> numa_node_map;

    _proc_type_table.assign((_numa_nodes == 1) ? 1 : _numa_nodes + 1, std::vector<int>({0, 0, 0, 0, -1, -1}));
    if (_numa_nodes > 1) {
        for (int i = 0; i < _numa_nodes; i++) {
            _proc_type_table[i + 1][PROC_NUMA_NODE_ID] = i;
        }
    } else {
        _proc_type_table[0][PROC_NUMA_NODE_ID] = 0;
    }
    if (_numa_nodes > 1) {
        for (int i = 1; i < static_cast<int>(_proc_type_table.size()); i++) {
            numa_node_map.insert(std::pair<int, int>(_proc_type_table[i][PROC_NUMA_NODE_ID], i));
        }
    } else {
        numa_node_map.insert(std::pair<int, int>(_proc_type_table[0][PROC_NUMA_NODE_ID], 0));
    }

    std::vector<int> all_table{0, 0, 0, 0, -1, -1};
    for (size_t i = 0; i < _cpu_mapping_table.size(); i++) {
        if (_cpu_mapping_table[i][CPU_MAP_USED_FLAG] == NOT_USED && _cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID] >= 0 &&
            _cpu_mapping_table[i][CPU_MAP_CORE_TYPE] >= ALL_PROC) {
            _proc_type_table[numa_node_map.at(_cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID])]
                            [_cpu_mapping_table[i][CPU_MAP_CORE_TYPE]]++;
            _proc_type_table[numa_node_map.at(_cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID])][ALL_PROC]++;
            _proc_type_table[numa_node_map.at(_cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID])][PROC_SOCKET_ID] =
                _cpu_mapping_table[i][CPU_MAP_SOCKET_ID];
            all_table[_cpu_mapping_table[i][CPU_MAP_CORE_TYPE]]++;
            all_table[ALL_PROC]++;
        }
    }
    if (_numa_nodes > 1) {
        _proc_type_table[0] = std::move(all_table);
    }

    if (_proc_type_table.size() > 1) {
        size_t n = _proc_type_table.size();

        while (n > 0) {
            if (0 == _proc_type_table[n - 1][ALL_PROC]) {
                _proc_type_table.erase(_proc_type_table.begin() + n - 1);
            }
            n--;
        }

        if ((_proc_type_table.size() > 1) && (_proc_type_table[0][ALL_PROC] == _proc_type_table[1][ALL_PROC])) {
            _proc_type_table.erase(_proc_type_table.begin());
        }
    }
}

}  // namespace threading
}  // namespace ov
