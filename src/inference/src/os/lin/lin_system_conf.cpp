// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sched.h>
#include <string.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "dev/threading/parallel_custom_arena.hpp"
#include "dev/threading/thread_affinity.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "os/cpu_map_info.hpp"

namespace ov {

CPU::CPU() {
    std::vector<std::vector<std::string>> system_info_table;
    std::vector<std::string> node_info_table;

    constexpr int cache_info_mode = 1;
    constexpr int freq_info_mode = 2;

    auto get_info_linux = [&](int mode) {
        int cpu_index = 0;
        int file_index = 0;
        int max_files = 3;

        std::string one_info;

        std::string::size_type pos = 0;
        std::string::size_type endpos = 0;
        std::string sub_str;

        int core_1;
        int core_2;

        system_info_table.clear();

        std::ifstream possible_file("/sys/devices/system/cpu/possible");
        std::string possible_info;

        if (possible_file.is_open()) {
            std::getline(possible_file, possible_info);
        } else {
            return -1;
        }

        if ((endpos = possible_info.find('-', pos)) != std::string::npos) {
            sub_str = possible_info.substr(pos, endpos - pos);
            core_1 = std::stoi(sub_str);
            sub_str = possible_info.substr(endpos + 1);
            core_2 = std::stoi(sub_str);
            system_info_table.resize(core_2 + 1, std::vector<std::string>(max_files, ""));
        } else {
            return -1;
        }

        std::ifstream online_file("/sys/devices/system/cpu/online");
        std::string online_info;

        if (online_file.is_open()) {
            std::getline(online_file, online_info);
        } else {
            system_info_table.clear();
            return -1;
        }

        while (1) {
            if ((endpos = online_info.find('-', pos)) != std::string::npos) {
                sub_str = online_info.substr(pos, endpos - pos);
                core_1 = std::stoi(sub_str);
                sub_str = online_info.substr(endpos + 1);
                core_2 = std::stoi(sub_str);

                for (cpu_index = core_1; cpu_index <= core_2; cpu_index++) {
                    if (mode == cache_info_mode) {
                        for (int n = 0; n < max_files; n++) {
                            file_index = (n == 0) ? n : n + 1;
                            one_info.clear();

                            std::ifstream cache_file("/sys/devices/system/cpu/cpu" + std::to_string(cpu_index) +
                                                     "/cache/index" + std::to_string(file_index) + "/shared_cpu_list");
                            if (cache_file.is_open()) {
                                std::getline(cache_file, one_info);
                            } else {
                                if ((cpu_index == core_1) && (n == 0)) {
                                    system_info_table.clear();
                                    return -1;
                                }
                            }
                            system_info_table[cpu_index][n] = std::move(one_info);
                        }
                    } else {
                        std::vector<std::string> file_name = {"/topology/core_cpus_list",
                                                              "/topology/physical_package_id",
                                                              "/cpufreq/cpuinfo_max_freq"};

                        for (int n = 0; n < max_files; n++) {
                            one_info.clear();

                            std::ifstream cache_file("/sys/devices/system/cpu/cpu" + std::to_string(cpu_index) +
                                                     file_name[n]);
                            if (cache_file.is_open()) {
                                std::getline(cache_file, one_info);
                            } else {
                                if ((cpu_index == core_1) && (n == 2)) {
                                    system_info_table.clear();
                                    return -1;
                                }
                            }
                            system_info_table[cpu_index][n] = std::move(one_info);
                        }
                    }
                }
            }

            if ((pos = online_info.find(',', endpos)) != std::string::npos) {
                pos++;
            } else {
                break;
            }
        }

        return 0;
    };

    auto get_node_info_linux = [&]() {
        int node_index = 0;

        while (1) {
            std::ifstream cache_file("/sys/devices/system/node/node" + std::to_string(node_index) + "/cpulist");
            if (!cache_file.is_open()) {
                break;
            }
            std::string cache_info;
            std::getline(cache_file, cache_info);
            if (cache_info.size() > 0) {
                node_info_table.emplace_back(std::move(cache_info));
            }
            node_index++;
        }
    };

    auto check_valid_cpu = [&]() {
        ov::threading::CpuSet mask;
        int ncpus = 0;
        std::tie(mask, ncpus) = ov::threading::get_process_mask();

        if ((_processors == 0) || mask == nullptr) {
            return -1;
        }

        std::vector<int> phy_core_list;
        std::vector<int> socket_list;
        std::vector<std::vector<int>> numa_node_list;
        std::vector<std::vector<int>> valid_cpu_mapping_table;

        numa_node_list.assign(_sockets, std::vector<int>());
        for (int i = 0; i < _processors; i++) {
            if (CPU_ISSET(i, mask)) {
                valid_cpu_mapping_table.emplace_back(_cpu_mapping_table[i]);
                if (_cpu_mapping_table[i][CPU_MAP_CORE_TYPE] == MAIN_CORE_PROC) {
                    phy_core_list.emplace_back(_cpu_mapping_table[i][CPU_MAP_CORE_ID]);
                }
                if (_sockets > 1) {
                    if (std::find(socket_list.begin(), socket_list.end(), _cpu_mapping_table[i][CPU_MAP_SOCKET_ID]) ==
                        socket_list.end()) {
                        socket_list.push_back(_cpu_mapping_table[i][CPU_MAP_SOCKET_ID]);
                    }
                    if (std::find(numa_node_list[_cpu_mapping_table[i][CPU_MAP_SOCKET_ID]].begin(),
                                  numa_node_list[_cpu_mapping_table[i][CPU_MAP_SOCKET_ID]].end(),
                                  _cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID]) ==
                        numa_node_list[_cpu_mapping_table[i][CPU_MAP_SOCKET_ID]].end()) {
                        numa_node_list[_cpu_mapping_table[i][CPU_MAP_SOCKET_ID]].push_back(
                            _cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID]);
                    }
                }
            }
        }
        if (_sockets > 1) {
            std::sort(socket_list.begin(), socket_list.end());
            for (int n = _sockets - 1; n >= 0; n--) {
                if (numa_node_list[n].size() == 0) {
                    numa_node_list.erase(numa_node_list.begin() + n);
                } else {
                    std::sort(numa_node_list[n].begin(), numa_node_list[n].end());
                }
            }
            std::map<int, int> sockets_map;
            std::map<int, int> numa_node_map;
            for (int i = 0; i < static_cast<int>(socket_list.size()); i++) {
                sockets_map.insert(std::pair<int, int>(socket_list[i], i));
            }
            for (int i = 0; i < static_cast<int>(numa_node_list.size()); i++) {
                for (int j = 0; j < static_cast<int>(numa_node_list[i].size()); j++) {
                    numa_node_map.insert(std::pair<int, int>(numa_node_list[i][j], i * _numa_nodes / _sockets + j));
                }
            }
            for (size_t i = 0; i < valid_cpu_mapping_table.size(); i++) {
                auto new_numa_id = numa_node_map.at(valid_cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID]);
                auto new_socket_id = sockets_map.at(valid_cpu_mapping_table[i][CPU_MAP_SOCKET_ID]);
                if (_numaid_mapping_table.find(new_numa_id) == _numaid_mapping_table.end()) {
                    _numaid_mapping_table.insert({new_numa_id, valid_cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID]});
                }
                if (_socketid_mapping_table.find(new_socket_id) == _socketid_mapping_table.end()) {
                    _socketid_mapping_table.insert({new_socket_id, valid_cpu_mapping_table[i][CPU_MAP_SOCKET_ID]});
                }
                valid_cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID] = new_numa_id;
                valid_cpu_mapping_table[i][CPU_MAP_SOCKET_ID] = new_socket_id;
            }
        }

        if (valid_cpu_mapping_table.size() == 0) {
            return -1;
        } else if (valid_cpu_mapping_table.size() == (unsigned)_processors) {
            return 0;
        } else {
            _processors = valid_cpu_mapping_table.size();
            _cpu_mapping_table.swap(valid_cpu_mapping_table);
            int cur_numa_nodes = _numa_nodes;
            int cur_cores = _cores;
            {
                std::lock_guard<std::mutex> lock{_cpu_mutex};
                update_valid_processor_linux(std::move(phy_core_list),
                                             cur_numa_nodes,
                                             cur_cores,
                                             _proc_type_table,
                                             _cpu_mapping_table);
            }
            _cores = cur_cores;
            _numa_nodes = cur_numa_nodes;
            return 0;
        }
    };

    get_node_info_linux();

    if (!get_info_linux(cache_info_mode)) {
        parse_cache_info_linux(system_info_table,
                               std::move(node_info_table),
                               _processors,
                               _numa_nodes,
                               _sockets,
                               _cores,
                               _proc_type_table,
                               _cpu_mapping_table);
    }

    if ((_proc_type_table.size() == 0) ||
        ((_proc_type_table[0][MAIN_CORE_PROC] == 0) && (_proc_type_table[0][ALL_PROC] > 0) &&
         (_proc_type_table[0][ALL_PROC] != _proc_type_table[0][EFFICIENT_CORE_PROC]))) {
        if (!get_info_linux(freq_info_mode)) {
            parse_freq_info_linux(system_info_table,
                                  std::move(node_info_table),
                                  _processors,
                                  _numa_nodes,
                                  _sockets,
                                  _cores,
                                  _proc_type_table,
                                  _cpu_mapping_table);
        }
    }

    if ((_proc_type_table.size() == 0) ||
        ((_proc_type_table[0][MAIN_CORE_PROC] == 0) && (_proc_type_table[0][ALL_PROC] > 0) &&
         (_proc_type_table[0][ALL_PROC] != _proc_type_table[0][EFFICIENT_CORE_PROC]))) {
        /*Previous CPU resource based on calculation*/
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::vector<int> processors;
        std::map<int, int> sockets;
        int socketId = 0;
        _cores = 0;
        while (!cpuinfo.eof()) {
            std::string line;
            std::getline(cpuinfo, line);
            if (line.empty())
                continue;
            auto delimeter = line.find(':');
            auto key = line.substr(0, delimeter);
            auto value = line.substr(delimeter + 1);
            if (0 == key.find("processor")) {
                processors.emplace_back(std::stoi(value));
            }
            if (0 == key.find("physical id")) {
                socketId = std::stoi(value);
            }
            if (0 == key.find("cpu cores")) {
                sockets[socketId] = std::stoi(value);
            }
        }
        _processors = processors.size();

        _numa_nodes = sockets.size() == 0 ? 1 : sockets.size();
        _sockets = _numa_nodes;

        for (auto&& socket : sockets) {
            _cores += socket.second;
        }
        if (_cores == 0) {
            _cores = _processors;
        }
        if (_processors > 0 && _numa_nodes > 0 && _cores > 0) {
            get_cpu_mapping_from_cores(_processors, _numa_nodes, _cores, _proc_type_table, _cpu_mapping_table);
        } else {
            OPENVINO_THROW("Wrong CPU information. processors: ",
                           _processors,
                           ", numa_nodes: ",
                           _numa_nodes,
                           ", cores: ",
                           _cores);
        }
    }
    std::vector<std::vector<std::string>>().swap(system_info_table);

    if (check_valid_cpu() < 0) {
        OPENVINO_THROW("CPU affinity check failed. No CPU is eligible to run inference.");
    };

    _org_proc_type_table = _proc_type_table;

    cpu_debug();
}

void parse_node_info_linux(const std::vector<std::string> node_info_table,
                           const int& _numa_nodes,
                           int& _sockets,
                           std::vector<std::vector<int>>& _proc_type_table,
                           std::vector<std::vector<int>>& _cpu_mapping_table) {
    std::vector<std::vector<int>> nodes_table;
    int node_index = 0;

    for (auto& one_info : node_info_table) {
        int core_1 = 0;
        int core_2 = 0;
        std::string::size_type pos = 0;
        std::string::size_type endpos = 0;
        std::string sub_str = "";

        if (((endpos = one_info.find('-', pos)) == std::string::npos) &&
            ((endpos = one_info.find(',', pos)) != std::string::npos)) {
            while (endpos != std::string::npos) {
                sub_str = one_info.substr(pos);
                core_1 = std::stoi(sub_str);
                nodes_table.push_back({core_1, core_1, node_index});
                endpos = one_info.find(',', pos);
                pos = endpos + 1;
            }
        } else {
            while (endpos != std::string::npos) {
                if ((endpos = one_info.find('-', pos)) != std::string::npos) {
                    sub_str = one_info.substr(pos, endpos - pos);
                    core_1 = std::stoi(sub_str);
                    sub_str = one_info.substr(endpos + 1);
                    core_2 = std::stoi(sub_str);
                    nodes_table.push_back({core_1, core_2, node_index});
                    pos = one_info.find(',', endpos);
                    if (pos == std::string::npos) {
                        break;
                    } else {
                        pos = pos + 1;
                    }
                }
            }
        }
        node_index++;
    }

    _proc_type_table.assign((node_info_table.size() == 1) ? 1 : node_info_table.size() + 1,
                            std::vector<int>({0, 0, 0, 0, -1, -1}));

    for (auto& row : nodes_table) {
        for (int i = row[0]; i <= row[1]; i++) {
            _cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID] = row[2];
            if (_sockets > _numa_nodes) {
                _cpu_mapping_table[i][CPU_MAP_SOCKET_ID] = row[2];
            }
            _proc_type_table[0][ALL_PROC]++;
            _proc_type_table[0][_cpu_mapping_table[i][CPU_MAP_CORE_TYPE]]++;
            if (node_info_table.size() != 1) {
                _proc_type_table[row[2] + 1][ALL_PROC]++;
                _proc_type_table[row[2] + 1][_cpu_mapping_table[i][CPU_MAP_CORE_TYPE]]++;
            }
        }
        node_index = (node_info_table.size() != 1) ? row[2] + 1 : 0;
        _proc_type_table[node_index][PROC_NUMA_NODE_ID] = _cpu_mapping_table[row[0]][CPU_MAP_NUMA_NODE_ID];
        _proc_type_table[node_index][PROC_SOCKET_ID] = _cpu_mapping_table[row[0]][CPU_MAP_SOCKET_ID];
    }
    _sockets = (_sockets > _numa_nodes) ? _numa_nodes : _sockets;
}

void parse_cache_info_linux(const std::vector<std::vector<std::string>> system_info_table,
                            const std::vector<std::string> node_info_table,
                            int& _processors,
                            int& _numa_nodes,
                            int& _sockets,
                            int& _cores,
                            std::vector<std::vector<int>>& _proc_type_table,
                            std::vector<std::vector<int>>& _cpu_mapping_table) {
    int n_group = 0;

    _processors = system_info_table.size();
    _cpu_mapping_table.resize(_processors, std::vector<int>(CPU_MAP_TABLE_SIZE, -1));

    auto clean_up_output = [&]() {
        _processors = 0;
        _cores = 0;
        _numa_nodes = 0;
        _sockets = 0;
        _cpu_mapping_table.clear();
        _proc_type_table.clear();
        return;
    };

    auto update_proc_map_info = [&](const int nproc) {
        if (-1 == _cpu_mapping_table[nproc][CPU_MAP_CORE_ID]) {
            int core_1 = 0;
            int core_2 = 0;
            std::string::size_type pos = 0;
            std::string::size_type endpos = 0;
            std::string sub_str = "";

            if (((endpos = system_info_table[nproc][0].find(',', pos)) != std::string::npos) ||
                ((endpos = system_info_table[nproc][0].find('-', pos)) != std::string::npos)) {
                sub_str = system_info_table[nproc][0].substr(pos, endpos - pos);
                core_1 = std::stoi(sub_str);
                sub_str = system_info_table[nproc][0].substr(endpos + 1);
                core_2 = std::stoi(sub_str);
                if ((core_1 != nproc) && (core_2 != nproc)) {
                    clean_up_output();
                    return;
                }

                _cpu_mapping_table[core_1][CPU_MAP_PROCESSOR_ID] = core_1;
                _cpu_mapping_table[core_2][CPU_MAP_PROCESSOR_ID] = core_2;

                _cpu_mapping_table[core_1][CPU_MAP_CORE_ID] = _cores;
                _cpu_mapping_table[core_2][CPU_MAP_CORE_ID] = _cores;

                /**
                 * Processor 0 need to handle system interception on Linux. So use second processor as physical core
                 * and first processor as logic core
                 */
                _cpu_mapping_table[core_1][CPU_MAP_CORE_TYPE] = HYPER_THREADING_PROC;
                _cpu_mapping_table[core_2][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;

                _cpu_mapping_table[core_1][CPU_MAP_GROUP_ID] = n_group;
                _cpu_mapping_table[core_2][CPU_MAP_GROUP_ID] = n_group;

                _cores++;

                _proc_type_table[0][ALL_PROC] += 2;
                _proc_type_table[0][MAIN_CORE_PROC]++;
                _proc_type_table[0][HYPER_THREADING_PROC]++;
            } else if ((endpos = system_info_table[nproc][1].find('-', pos)) != std::string::npos) {
                sub_str = system_info_table[nproc][1].substr(pos, endpos - pos);
                core_1 = std::stoi(sub_str);
                sub_str = system_info_table[nproc][1].substr(endpos + 1);
                core_2 = std::stoi(sub_str);
                if ((core_2 - core_1 == 1) && (_proc_type_table[0][EFFICIENT_CORE_PROC] == 0)) {
                    _cpu_mapping_table[core_1][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
                } else {
                    _cpu_mapping_table[core_1][CPU_MAP_CORE_TYPE] = EFFICIENT_CORE_PROC;
                }

                for (int m = core_1; m <= core_2; m++) {
                    _cpu_mapping_table[m][CPU_MAP_PROCESSOR_ID] = m;
                    _cpu_mapping_table[m][CPU_MAP_CORE_ID] = _cores;
                    _cpu_mapping_table[m][CPU_MAP_CORE_TYPE] = _cpu_mapping_table[core_1][CPU_MAP_CORE_TYPE];
                    _cpu_mapping_table[m][CPU_MAP_GROUP_ID] = n_group;

                    _cores++;

                    _proc_type_table[0][ALL_PROC]++;
                    _proc_type_table[0][_cpu_mapping_table[m][CPU_MAP_CORE_TYPE]]++;
                }
            } else {
                core_1 = std::stoi(system_info_table[nproc][0]);

                _cpu_mapping_table[core_1][CPU_MAP_PROCESSOR_ID] = core_1;
                _cpu_mapping_table[core_1][CPU_MAP_CORE_ID] = _cores;
                _cpu_mapping_table[core_1][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
                _cpu_mapping_table[core_1][CPU_MAP_GROUP_ID] = n_group;

                _cores++;

                _proc_type_table[0][ALL_PROC]++;
                _proc_type_table[0][MAIN_CORE_PROC]++;
            }

            n_group++;
            _proc_type_table[0][PROC_NUMA_NODE_ID] = (_proc_type_table[0][PROC_NUMA_NODE_ID] == -1)
                                                         ? _cpu_mapping_table[core_1][CPU_MAP_NUMA_NODE_ID]
                                                         : _proc_type_table[0][PROC_NUMA_NODE_ID];
            _proc_type_table[0][PROC_SOCKET_ID] = (_proc_type_table[0][PROC_SOCKET_ID] == -1)
                                                      ? _cpu_mapping_table[core_1][CPU_MAP_SOCKET_ID]
                                                      : _proc_type_table[0][PROC_SOCKET_ID];
        }
        return;
    };

    const std::vector<int> line_value_0({0, 0, 0, 0, -1, -1});

    std::vector<int> offline_list;
    int info_index = 0;

    for (int n = 0; n < _processors; n++) {
        if ((system_info_table[n][2].size() > 0) || (system_info_table[n][1].size() > 0)) {
            info_index = system_info_table[n][2].size() > 0 ? 2 : 1;
            if (-1 == _cpu_mapping_table[n][CPU_MAP_SOCKET_ID]) {
                std::string::size_type pos = 0, endpos = 0, endpos1 = 0;
                std::string sub_str;

                int core_1;
                int core_2;

                if (0 == _sockets) {
                    _proc_type_table.push_back(line_value_0);
                } else {
                    _proc_type_table.push_back(_proc_type_table[0]);
                    _proc_type_table[0] = line_value_0;
                }

                while (1) {
                    endpos = system_info_table[n][info_index].find('-', pos);
                    endpos1 = system_info_table[n][info_index].find(',', pos);

                    if (endpos < endpos1) {
                        sub_str = system_info_table[n][info_index].substr(pos, endpos - pos);
                        core_1 = std::stoi(sub_str);
                        sub_str = system_info_table[n][info_index].substr(endpos + 1);
                        core_2 = std::stoi(sub_str);

                        if ((info_index == 1) && (core_2 - core_1 == 1)) {
                            offline_list.push_back(n);
                            break;
                        }
                        for (int m = core_1; m <= core_2; m++) {
                            _cpu_mapping_table[m][CPU_MAP_SOCKET_ID] = _sockets;
                            _cpu_mapping_table[m][CPU_MAP_NUMA_NODE_ID] = _cpu_mapping_table[m][CPU_MAP_SOCKET_ID];
                            update_proc_map_info(m);
                            if (_processors == 0) {
                                return;
                            };
                        }
                    } else {
                        sub_str = system_info_table[n][info_index].substr(pos, endpos1 - pos);
                        core_1 = std::stoi(sub_str);
                        _cpu_mapping_table[core_1][CPU_MAP_SOCKET_ID] = _sockets;
                        _cpu_mapping_table[core_1][CPU_MAP_NUMA_NODE_ID] =
                            _cpu_mapping_table[core_1][CPU_MAP_SOCKET_ID];
                        update_proc_map_info(core_1);
                        if (_processors == 0) {
                            return;
                        };
                    }

                    if (endpos1 != std::string::npos) {
                        pos = endpos1 + 1;
                    } else {
                        break;
                    }
                }
                _sockets++;
                if (_proc_type_table[0][ALL_PROC] == 0) {
                    _proc_type_table.erase(_proc_type_table.begin());
                    _sockets--;
                }
            }
        } else {
            offline_list.push_back(n);
        }
    }

    if ((node_info_table.size() == 0) || (node_info_table.size() == (unsigned)_sockets)) {
        if (_sockets > 1) {
            _proc_type_table.push_back(_proc_type_table[0]);
            _proc_type_table[0] = line_value_0;

            for (int m = 1; m <= _sockets; m++) {
                for (int n = 0; n < PROC_NUMA_NODE_ID; n++) {
                    _proc_type_table[0][n] += _proc_type_table[m][n];
                }
            }
        }
        _numa_nodes = _sockets;
    } else {
        _numa_nodes = node_info_table.size();
        parse_node_info_linux(node_info_table, _numa_nodes, _sockets, _proc_type_table, _cpu_mapping_table);
    }

    for (size_t n = 0; n < offline_list.size(); n++) {
        _cpu_mapping_table.erase(_cpu_mapping_table.begin() + offline_list[n] - n);
        _processors--;
    }
};

void get_cpu_mapping_from_cores(const int _processors,
                                const int _numa_nodes,
                                const int _cores,
                                std::vector<std::vector<int>>& _proc_type_table,
                                std::vector<std::vector<int>>& _cpu_mapping_table) {
    const auto hyper_thread = _processors > _cores ? true : false;
    const auto num_big_cores = hyper_thread ? (_processors - _cores) * 2 : _cores;
    int big_phys_cores = hyper_thread ? num_big_cores / 2 : num_big_cores;
    const auto num_small_cores_phys = _processors - num_big_cores;
    const auto socket_offset = big_phys_cores / _numa_nodes;
    const auto threads_per_core = hyper_thread ? 2 : 1;
    const auto step = num_small_cores_phys > 0 ? 2 : 1;
    std::vector<int> pro_all_table = {0, 0, 0, 0, -1, -1};

    _cpu_mapping_table.resize(_processors, std::vector<int>(CPU_MAP_TABLE_SIZE, -1));
    _proc_type_table.assign(_numa_nodes, std::vector<int>({0, 0, 0, 0, -1, -1}));

    for (int t = 0; t < threads_per_core; t++) {
        int start = t == 0 ? 0 : (num_small_cores_phys > 0 ? 1 : big_phys_cores);
        for (int i = 0; i < big_phys_cores; i++) {
            int socket_id = _numa_nodes > 1 ? i / socket_offset : 0;
            int cur_id = start + i * step;
            _cpu_mapping_table[cur_id][CPU_MAP_PROCESSOR_ID] = cur_id;
            _cpu_mapping_table[cur_id][CPU_MAP_CORE_ID] = i;
            _cpu_mapping_table[cur_id][CPU_MAP_CORE_TYPE] =
                hyper_thread ? (t == 0 ? HYPER_THREADING_PROC : MAIN_CORE_PROC) : MAIN_CORE_PROC;
            _cpu_mapping_table[cur_id][CPU_MAP_GROUP_ID] = i;
            _cpu_mapping_table[cur_id][CPU_MAP_NUMA_NODE_ID] = socket_id;
            _cpu_mapping_table[cur_id][CPU_MAP_SOCKET_ID] = socket_id;

            _proc_type_table[socket_id][_cpu_mapping_table[cur_id][CPU_MAP_CORE_TYPE]]++;
            _proc_type_table[socket_id][ALL_PROC]++;
            _proc_type_table[socket_id][PROC_NUMA_NODE_ID] = (_proc_type_table[socket_id][PROC_NUMA_NODE_ID] == -1)
                                                                 ? socket_id
                                                                 : _proc_type_table[socket_id][PROC_NUMA_NODE_ID];
            _proc_type_table[socket_id][PROC_SOCKET_ID] = (_proc_type_table[socket_id][PROC_SOCKET_ID] == -1)
                                                              ? socket_id
                                                              : _proc_type_table[socket_id][PROC_SOCKET_ID];
            pro_all_table[_cpu_mapping_table[cur_id][CPU_MAP_CORE_TYPE]]++;
            pro_all_table[ALL_PROC]++;
        }
    }
    if (num_small_cores_phys > 0) {
        for (int j = 0; j < num_small_cores_phys; j++) {
            int cur_id = num_big_cores + j;
            _cpu_mapping_table[cur_id][CPU_MAP_PROCESSOR_ID] = cur_id;
            _cpu_mapping_table[cur_id][CPU_MAP_CORE_ID] = big_phys_cores + j;
            _cpu_mapping_table[cur_id][CPU_MAP_CORE_TYPE] = EFFICIENT_CORE_PROC;
            _cpu_mapping_table[cur_id][CPU_MAP_GROUP_ID] = big_phys_cores + j / 4;
            _cpu_mapping_table[cur_id][CPU_MAP_NUMA_NODE_ID] = 0;
            _cpu_mapping_table[cur_id][CPU_MAP_SOCKET_ID] = 0;

            _proc_type_table[0][_cpu_mapping_table[cur_id][CPU_MAP_CORE_TYPE]]++;
            _proc_type_table[0][ALL_PROC]++;
            pro_all_table[_cpu_mapping_table[cur_id][CPU_MAP_CORE_TYPE]]++;
            pro_all_table[ALL_PROC]++;
        }
    }
    if (_numa_nodes > 1) {
        _proc_type_table.insert(_proc_type_table.begin(), pro_all_table);
    }
}

void parse_freq_info_linux(const std::vector<std::vector<std::string>> system_info_table,
                           const std::vector<std::string> node_info_table,
                           int& _processors,
                           int& _numa_nodes,
                           int& _sockets,
                           int& _cores,
                           std::vector<std::vector<int>>& _proc_type_table,
                           std::vector<std::vector<int>>& _cpu_mapping_table) {
    int freq_max = 0;
    bool ecore_enabled = false;

    _processors = system_info_table.size();
    _numa_nodes = 0;
    _sockets = 0;
    _cores = 0;
    _cpu_mapping_table.resize(_processors, std::vector<int>(CPU_MAP_TABLE_SIZE, -1));

    std::vector<int> line_value_0(PROC_TYPE_TABLE_SIZE, 0);

    std::vector<int> offline_list;

    auto clean_up_output = [&]() {
        _processors = 0;
        _cores = 0;
        _numa_nodes = 0;
        _sockets = 0;
        _cpu_mapping_table.clear();
        _proc_type_table.clear();
        return;
    };

    for (int n = 0; n < _processors; n++) {
        if (system_info_table[n][2].size() > 0) {
            if (-1 == _cpu_mapping_table[n][CPU_MAP_SOCKET_ID]) {
                std::string::size_type pos = 0;
                std::string::size_type endpos1 = 0;
                std::string::size_type endpos2 = 0;
                std::string sub_str;

                int core_1 = 0;
                int core_2 = 0;

                if (((endpos1 = system_info_table[n][0].find(',', pos)) != std::string::npos) ||
                    ((endpos2 = system_info_table[n][0].find('-', pos)) != std::string::npos)) {
                    endpos1 = (endpos1 != std::string::npos) ? endpos1 : endpos2;
                    sub_str = system_info_table[n][0].substr(pos, endpos1 - pos);
                    core_1 = std::stoi(sub_str);
                    sub_str = system_info_table[n][0].substr(endpos1 + 1);
                    core_2 = std::stoi(sub_str);
                    if ((core_1 != n) && (core_2 != n)) {
                        clean_up_output();
                        return;
                    }

                    _cpu_mapping_table[core_1][CPU_MAP_PROCESSOR_ID] = core_1;
                    _cpu_mapping_table[core_1][CPU_MAP_SOCKET_ID] = std::stoi(system_info_table[core_1][1]);
                    _cpu_mapping_table[core_1][CPU_MAP_NUMA_NODE_ID] = _cpu_mapping_table[core_1][CPU_MAP_SOCKET_ID];
                    _cpu_mapping_table[core_1][CPU_MAP_CORE_ID] = _cores;
                    _cpu_mapping_table[core_1][CPU_MAP_CORE_TYPE] = HYPER_THREADING_PROC;
                    _cpu_mapping_table[core_1][CPU_MAP_GROUP_ID] = _cores;

                    _cpu_mapping_table[core_2][CPU_MAP_PROCESSOR_ID] = core_2;
                    _cpu_mapping_table[core_2][CPU_MAP_SOCKET_ID] = _cpu_mapping_table[core_1][CPU_MAP_SOCKET_ID];
                    _cpu_mapping_table[core_2][CPU_MAP_NUMA_NODE_ID] = _cpu_mapping_table[core_1][CPU_MAP_SOCKET_ID];
                    _cpu_mapping_table[core_2][CPU_MAP_CORE_ID] = _cpu_mapping_table[core_1][CPU_MAP_CORE_ID];
                    _cpu_mapping_table[core_2][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
                    _cpu_mapping_table[core_2][CPU_MAP_GROUP_ID] = _cpu_mapping_table[core_1][CPU_MAP_GROUP_ID];

                    int core_freq = std::stoi(system_info_table[core_1][2]);
                    freq_max = std::max(core_freq, freq_max);
                } else if (system_info_table[n][0].size() > 0) {
                    core_1 = std::stoi(system_info_table[n][0]);

                    _cpu_mapping_table[core_1][CPU_MAP_PROCESSOR_ID] = core_1;
                    _cpu_mapping_table[core_1][CPU_MAP_SOCKET_ID] = std::stoi(system_info_table[core_1][1]);
                    _cpu_mapping_table[core_1][CPU_MAP_NUMA_NODE_ID] = _cpu_mapping_table[core_1][CPU_MAP_SOCKET_ID];
                    _cpu_mapping_table[core_1][CPU_MAP_CORE_ID] = _cores;

                    int core_freq = std::stoi(system_info_table[core_1][2]);
                    if ((0 == freq_max) || (core_freq >= freq_max * 0.97)) {
                        freq_max = std::max(core_freq, freq_max);
                        _cpu_mapping_table[core_1][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
                    } else {
                        _cpu_mapping_table[core_1][CPU_MAP_CORE_TYPE] = EFFICIENT_CORE_PROC;
                        ecore_enabled = true;
                    }

                    _cpu_mapping_table[core_1][CPU_MAP_GROUP_ID] = _cores;
                }
                _sockets = std::max(_sockets, _cpu_mapping_table[core_1][CPU_MAP_SOCKET_ID]);
                _cores++;
            }
        } else {
            offline_list.push_back(n);
        }
    }

    _sockets = (_sockets > 0) ? _sockets + 1 : 1;

    if (node_info_table.size() == 0) {
        if ((_sockets > 1) && (ecore_enabled)) {
            _sockets = 1;  // This is the WA of the developing platform without CPU cache and numa node information.
                           // Wrong socket information creates each socket ID per CPU core.
        }
        if (_sockets > 1) {
            _proc_type_table.resize(_sockets + 1, std::vector<int>({0, 0, 0, 0, -1, -1}));
            for (int n = 0; n < _processors; n++) {
                _proc_type_table[0][ALL_PROC]++;
                _proc_type_table[_cpu_mapping_table[n][CPU_MAP_SOCKET_ID] + 1][ALL_PROC]++;

                _proc_type_table[0][_cpu_mapping_table[n][CPU_MAP_CORE_TYPE]]++;
                _proc_type_table[_cpu_mapping_table[n][CPU_MAP_SOCKET_ID] + 1]
                                [_cpu_mapping_table[n][CPU_MAP_CORE_TYPE]]++;
            }
            for (int n = 0; n < _sockets; n++) {
                _proc_type_table[n + 1][PROC_NUMA_NODE_ID] = n;
                _proc_type_table[n + 1][PROC_SOCKET_ID] = n;
            };
        } else {
            _proc_type_table.resize(1, std::vector<int>({0, 0, 0, 0, 0, 0}));
            for (int n = 0; n < _processors; n++) {
                _proc_type_table[0][ALL_PROC]++;
                _proc_type_table[0][_cpu_mapping_table[n][CPU_MAP_CORE_TYPE]]++;
                _cpu_mapping_table[n][CPU_MAP_NUMA_NODE_ID] = 0;
                _cpu_mapping_table[n][CPU_MAP_SOCKET_ID] = 0;
            }
        }
        _numa_nodes = _sockets;
    } else {
        _numa_nodes = node_info_table.size();
        parse_node_info_linux(node_info_table, _numa_nodes, _sockets, _proc_type_table, _cpu_mapping_table);
    }

    for (size_t n = 0; n < offline_list.size(); n++) {
        _cpu_mapping_table.erase(_cpu_mapping_table.begin() + offline_list[n] - n);
        _processors--;
    }
};

void update_valid_processor_linux(const std::vector<int> phy_core_list,
                                  int& _sockets,
                                  int& _cores,
                                  std::vector<std::vector<int>>& _proc_type_table,
                                  std::vector<std::vector<int>>& _cpu_mapping_table) {
    for (auto& row : _proc_type_table) {
        std::fill(row.begin(), row.begin() + PROC_NUMA_NODE_ID, 0);
    }
    _cores = 0;
    for (auto& row : _cpu_mapping_table) {
        if (row[CPU_MAP_CORE_TYPE] == HYPER_THREADING_PROC) {
            auto iter = std::find(phy_core_list.begin(), phy_core_list.end(), row[CPU_MAP_CORE_ID]);
            if (iter == phy_core_list.end()) {
                row[CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
                _cores++;
            }
        } else {
            _cores++;
        }

        _proc_type_table[0][ALL_PROC]++;
        _proc_type_table[0][row[CPU_MAP_CORE_TYPE]]++;
        if (_proc_type_table.size() > 1) {
            _proc_type_table[row[CPU_MAP_NUMA_NODE_ID] + 1][ALL_PROC]++;
            _proc_type_table[row[CPU_MAP_NUMA_NODE_ID] + 1][row[CPU_MAP_CORE_TYPE]]++;
        }
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
    _sockets = _proc_type_table.size() == 1 ? 1 : _proc_type_table.size() - 1;
    return;
};

}  // namespace ov
