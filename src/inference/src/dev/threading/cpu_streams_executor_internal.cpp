// Copyright (C) 2018-2023 Intel Corporation
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
                         const bool cpu_reservation,
                         const std::vector<std::vector<int>> proc_type_table,
                         const std::vector<std::vector<int>> streams_info_table,
                         StreamCreateType& stream_type,
                         int& concurrency,
                         int& core_type,
                         int& numa_node_id,
                         int& max_threads_per_core) {
    int stream_total = 0;
    size_t stream_info_id = 0;
    bool cpu_reserve = cpu_reservation;
    for (size_t i = 0; i < streams_info_table.size(); i++) {
        stream_total += streams_info_table[i][NUMBER_OF_STREAMS];
        if (stream_id < stream_total) {
            stream_info_id = i;
            break;
        }
    }
    concurrency = streams_info_table[stream_info_id][THREADS_PER_STREAM];
    core_type = streams_info_table[stream_info_id][PROC_TYPE];
    numa_node_id = streams_info_table[stream_info_id][STREAM_NUMA_NODE_ID];
    max_threads_per_core = 1;
    bool have_main_core_proc = false;
    bool have_hyper_threading_proc = false;
    if (core_type == ALL_PROC) {
        for (size_t i = stream_info_id + 1; i < streams_info_table.size(); i++) {
            if (streams_info_table[i][NUMBER_OF_STREAMS] == 0) {
                if (streams_info_table[i][PROC_TYPE] == HYPER_THREADING_PROC) {
                    have_hyper_threading_proc = true;
                } else if (streams_info_table[i][PROC_TYPE] == MAIN_CORE_PROC) {
                    have_main_core_proc = true;
                }
            } else {
                break;
            }
        }
    }

    if (have_main_core_proc && have_hyper_threading_proc) {
        max_threads_per_core = 2;
    }

#if defined(_WIN32) || defined(__APPLE__)
    cpu_reserve = false;
#endif
    if (cpu_reserve) {
        stream_type = STREAM_WITH_OBSERVE;
    } else {
        stream_type = STREAM_WITHOUT_PARAM;
        if (proc_type_table[0][EFFICIENT_CORE_PROC] > 0 && core_type != ALL_PROC) {
            stream_type = STREAM_WITH_CORE_TYPE;
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
    std::vector<std::vector<int>> streams_info_table = _streams_info_table;
    std::vector<std::vector<std::string>> stream_conditions;
    std::vector<int> stream_pos;
    std::vector<int> stream_num;
    int num_streams = 0;

    stream_pos.assign(_streams_info_table.size(), 0);
    stream_num.assign(_streams_info_table.size(), 0);
    for (size_t i = 0; i < _streams_info_table.size(); i++) {
        stream_pos[i] = num_streams;
        num_streams += _streams_info_table[i][NUMBER_OF_STREAMS];
    }
    _stream_processors.assign(num_streams, std::vector<int>());
    stream_conditions.assign(_streams_info_table.size(), std::vector<std::string>());
    for (size_t i = 0; i < _streams_info_table.size(); i++) {
        std::vector<std::string> proc_types;
        std::vector<std::string> numa_nodes;
        std::vector<std::string> sockets;
        if (_streams_info_table[i][PROC_TYPE] > ALL_PROC && _streams_info_table[i][NUMBER_OF_STREAMS] > 0) {
            proc_types.push_back(std::to_string(_streams_info_table[i][PROC_TYPE]));
        }
        if (num_streams == 1 && _streams_info_table[0][PROC_TYPE] == MAIN_CORE_PROC &&
            _streams_info_table[0][THREADS_PER_STREAM] > _proc_type_table[0][MAIN_CORE_PROC]) {
            proc_types.push_back(std::to_string(HYPER_THREADING_PROC));
        }
        if (_streams_info_table[i][STREAM_NUMA_NODE_ID] < 0) {
            for (int j = 0; j < _numa_nodes; j++) {
                numa_nodes.push_back(std::to_string(j));
            }
        } else {
            numa_nodes.push_back(std::to_string(_streams_info_table[i][STREAM_NUMA_NODE_ID]));
        }
        if (_streams_info_table[i][STREAM_SOCKET_ID] < 0) {
            for (int j = 0; j < _numa_nodes; j++) {
                sockets.push_back(std::to_string(j));
            }
        } else {
            sockets.push_back(std::to_string(_streams_info_table[i][STREAM_SOCKET_ID]));
        }
        for (auto t : proc_types) {
            for (auto n : numa_nodes) {
                for (auto s : sockets) {
                    stream_conditions[i].push_back(t + n + s);
                }
            }
        }
    }

    for (size_t i = 0; i < _cpu_mapping_table.size(); i++) {
        std::string cpu_string = std::to_string(_cpu_mapping_table[i][CPU_MAP_CORE_TYPE]) +
                                 std::to_string(_cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID]) +
                                 std::to_string(_cpu_mapping_table[i][CPU_MAP_SOCKET_ID]);
        for (size_t j = 0; j < stream_conditions.size(); j++) {
            if (std::find(stream_conditions[j].begin(), stream_conditions[j].end(), cpu_string) !=
                stream_conditions[j].end()) {
                _stream_processors[stream_pos[j]].push_back(_cpu_mapping_table[i][CPU_MAP_PROCESSOR_ID]);
                _cpu_mapping_table[i][CPU_MAP_USED_FLAG] = _cpu_status;
                if (static_cast<int>(_stream_processors[stream_pos[j]].size()) ==
                    _streams_info_table[j][THREADS_PER_STREAM]) {
                    stream_pos[j]++;
                    stream_num[j]++;
                }
                if (stream_num[j] >= _streams_info_table[j][NUMBER_OF_STREAMS]) {
                    stream_conditions[j].clear();
                }
                break;
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
    std::vector<int> all_table;
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
    all_table = {0, 0, 0, 0, -1, -1};
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
        _proc_type_table[0] = all_table;
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
