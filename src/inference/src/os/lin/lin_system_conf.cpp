// Copyright (C) 2018-2023 Intel Corporation
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
#include "openvino/core/except.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "streams_executor.hpp"

namespace ov {

CPU::CPU() {
    std::vector<std::vector<std::string>> system_info_table;

    _num_threads = parallel_get_max_threads();
    auto GetCacheInfoLinux = [&]() {
        int cpu_index = 0;
        int cache_index = 0;
        int cache_files = 3;

        std::vector<std::string> one_info(cache_files);

        while (1) {
            for (int n = 0; n < cache_files; n++) {
                cache_index = (n == 0) ? n : n + 1;

                std::ifstream cache_file("/sys/devices/system/cpu/cpu" + std::to_string(cpu_index) + "/cache/index" +
                                         std::to_string(cache_index) + "/shared_cpu_list");
                if (!cache_file.is_open()) {
                    cache_index = -1;
                    break;
                }
                std::string cache_info;
                std::getline(cache_file, cache_info);
                one_info[n] = cache_info;
            }

            if (cache_index == -1) {
                if (cpu_index == 0) {
                    return -1;
                } else {
                    return 0;
                }
            } else {
                system_info_table.push_back(one_info);
                cpu_index++;
            }
        }

        return 0;
    };

    auto GetFreqInfoLinux = [&]() {
        int cpu_index = 0;
        int cache_index = 0;

        std::vector<std::string> file_name = {"/topology/core_cpus_list",
                                              "/topology/physical_package_id",
                                              "/cpufreq/cpuinfo_max_freq"};
        int num_of_files = file_name.size();
        std::vector<std::string> one_info(num_of_files);

        while (1) {
            for (int n = 0; n < num_of_files; n++) {
                cache_index = n;

                std::ifstream cache_file("/sys/devices/system/cpu/cpu" + std::to_string(cpu_index) + file_name[n]);
                if (!cache_file.is_open()) {
                    cache_index = -1;
                    break;
                }
                std::string cache_info;
                std::getline(cache_file, cache_info);
                one_info[n] = cache_info;
            }

            if (cache_index == -1) {
                if (cpu_index == 0) {
                    return -1;
                } else {
                    return 0;
                }
            } else {
                system_info_table.push_back(one_info);
                cpu_index++;
            }
        }

        return 0;
    };

    if (!GetCacheInfoLinux()) {
        parse_cache_info_linux(system_info_table,
                               _processors,
                               _numa_nodes,
                               _cores,
                               _proc_type_table,
                               _cpu_mapping_table);
    }

    if ((_proc_type_table.size() == 0) || (_proc_type_table[0][MAIN_CORE_PROC] == 0)) {
        if (!GetFreqInfoLinux()) {
            parse_freq_info_linux(system_info_table,
                                  _processors,
                                  _numa_nodes,
                                  _cores,
                                  _proc_type_table,
                                  _cpu_mapping_table);
        }
    }

    if ((_proc_type_table.size() == 0) || (_proc_type_table[0][MAIN_CORE_PROC] == 0)) {
        /*Previous CPU resource based on calculation*/
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::vector<int> processors;
        std::map<int, int> sockets;
        int socketId = 0;
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
        _numa_nodes = sockets.size();
        for (auto&& socket : sockets) {
            _cores += socket.second;
        }
        if (_cores == 0) {
            _cores = _processors;
        }
    }
    std::vector<std::vector<std::string>>().swap(system_info_table);
}

void parse_cache_info_linux(const std::vector<std::vector<std::string>> system_info_table,
                            int& _processors,
                            int& _sockets,
                            int& _cores,
                            std::vector<std::vector<int>>& _proc_type_table,
                            std::vector<std::vector<int>>& _cpu_mapping_table) {
    int n_group = 0;

    _processors = system_info_table.size();
    _cpu_mapping_table.resize(_processors, std::vector<int>(CPU_MAP_TABLE_SIZE, -1));

    auto UpdateProcMapping = [&](const int nproc) {
        if (-1 == _cpu_mapping_table[nproc][CPU_MAP_CORE_ID]) {
            int core_1 = 0;
            int core_2 = 0;
            std::string::size_type pos = 0;
            std::string::size_type endpos = 0;
            std::string sub_str = "";

            if (((endpos = system_info_table[nproc][0].find(',', pos)) != std::string::npos) ||
                ((endpos = system_info_table[nproc][0].find('-', pos)) != std::string::npos)) {
                sub_str = system_info_table[nproc][0].substr(pos, endpos);
                core_1 = std::stoi(sub_str);
                sub_str = system_info_table[nproc][0].substr(endpos + 1);
                core_2 = std::stoi(sub_str);

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
                n_group++;

                _proc_type_table[0][ALL_PROC] += 2;
                _proc_type_table[0][MAIN_CORE_PROC]++;
                _proc_type_table[0][HYPER_THREADING_PROC]++;
            } else if ((endpos = system_info_table[nproc][1].find('-', pos)) != std::string::npos) {
                sub_str = system_info_table[nproc][1].substr(pos, endpos);
                core_1 = std::stoi(sub_str);
                sub_str = system_info_table[nproc][1].substr(endpos + 1);
                core_2 = std::stoi(sub_str);

                for (int m = core_1; m <= core_2; m++) {
                    _cpu_mapping_table[m][CPU_MAP_PROCESSOR_ID] = m;
                    _cpu_mapping_table[m][CPU_MAP_CORE_ID] = _cores;
                    _cpu_mapping_table[m][CPU_MAP_CORE_TYPE] = EFFICIENT_CORE_PROC;
                    _cpu_mapping_table[m][CPU_MAP_GROUP_ID] = n_group;

                    _cores++;

                    _proc_type_table[0][ALL_PROC]++;
                    _proc_type_table[0][EFFICIENT_CORE_PROC]++;
                }

                n_group++;
            } else {
                core_1 = std::stoi(system_info_table[nproc][0]);

                _cpu_mapping_table[core_1][CPU_MAP_PROCESSOR_ID] = core_1;
                _cpu_mapping_table[core_1][CPU_MAP_CORE_ID] = _cores;
                _cpu_mapping_table[core_1][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
                _cpu_mapping_table[core_1][CPU_MAP_GROUP_ID] = n_group;

                _cores++;
                n_group++;

                _proc_type_table[0][ALL_PROC]++;
                _proc_type_table[0][MAIN_CORE_PROC]++;
            }
        }
        return;
    };

    std::vector<int> line_value_0(PROC_TYPE_TABLE_SIZE, 0);

    for (int n = 0; n < _processors; n++) {
        if (-1 == _cpu_mapping_table[n][CPU_MAP_SOCKET_ID]) {
            std::string::size_type pos = 0;
            std::string::size_type endpos = 0;
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
                if ((endpos = system_info_table[n][2].find('-', pos)) != std::string::npos) {
                    sub_str = system_info_table[n][2].substr(pos, endpos);
                    core_1 = std::stoi(sub_str);
                    sub_str = system_info_table[n][2].substr(endpos + 1);
                    core_2 = std::stoi(sub_str);

                    for (int m = core_1; m <= core_2; m++) {
                        _cpu_mapping_table[m][CPU_MAP_SOCKET_ID] = _sockets;
                        UpdateProcMapping(m);
                    }
                } else if (pos != std::string::npos) {
                    sub_str = system_info_table[n][2].substr(pos);
                    core_1 = std::stoi(sub_str);
                    _cpu_mapping_table[core_1][CPU_MAP_SOCKET_ID] = _sockets;
                    UpdateProcMapping(core_1);
                    endpos = pos;
                }

                if ((pos = system_info_table[n][2].find(',', endpos)) != std::string::npos) {
                    pos++;
                } else {
                    break;
                }
            }
            _sockets++;
        }
    }
    if (_sockets > 1) {
        _proc_type_table.push_back(_proc_type_table[0]);
        _proc_type_table[0] = line_value_0;

        for (int m = 1; m <= _sockets; m++) {
            for (int n = 0; n < PROC_TYPE_TABLE_SIZE; n++) {
                _proc_type_table[0][n] += _proc_type_table[m][n];
            }
        }
    }
};

void parse_freq_info_linux(const std::vector<std::vector<std::string>> system_info_table,
                           int& _processors,
                           int& _sockets,
                           int& _cores,
                           std::vector<std::vector<int>>& _proc_type_table,
                           std::vector<std::vector<int>>& _cpu_mapping_table) {
    std::vector<int> freq_list = {0, 0, 0, 0};

    _processors = system_info_table.size();
    _cpu_mapping_table.resize(_processors, std::vector<int>(CPU_MAP_TABLE_SIZE, -1));

    auto UpdateProcMapping = [&](const int nproc) {
        _cpu_mapping_table[nproc][CPU_MAP_PROCESSOR_ID] = nproc;
        _cpu_mapping_table[nproc][CPU_MAP_SOCKET_ID] = std::stoi(system_info_table[nproc][1]);
        _cpu_mapping_table[nproc][CPU_MAP_CORE_ID] = _cores;

        _sockets = std::max(_sockets, _cpu_mapping_table[nproc][CPU_MAP_SOCKET_ID]);

        int core_freq = std::stoi(system_info_table[nproc][2]);
        if ((0 == freq_list[MAIN_CORE_PROC]) || (core_freq == freq_list[MAIN_CORE_PROC])) {
            freq_list[MAIN_CORE_PROC] = core_freq;
            _cpu_mapping_table[nproc][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
        } else if ((0 == freq_list[EFFICIENT_CORE_PROC]) || (core_freq == freq_list[EFFICIENT_CORE_PROC])) {
            freq_list[EFFICIENT_CORE_PROC] = core_freq;
            _cpu_mapping_table[nproc][CPU_MAP_CORE_TYPE] = EFFICIENT_CORE_PROC;
        }

        _cpu_mapping_table[nproc][CPU_MAP_GROUP_ID] = _cores;

        return;
    };

    std::vector<int> line_value_0(PROC_TYPE_TABLE_SIZE, 0);

    for (int n = 0; n < _processors; n++) {
        if (-1 == _cpu_mapping_table[n][CPU_MAP_SOCKET_ID]) {
            std::string::size_type pos = 0;
            std::string::size_type endpos1 = 0;
            std::string::size_type endpos2 = 0;
            std::string sub_str;

            int core_1;
            int core_2;

            if (((endpos1 = system_info_table[n][0].find(',', pos)) != std::string::npos) ||
                ((endpos2 = system_info_table[n][0].find('-', pos)) != std::string::npos)) {
                endpos1 = (endpos1 != std::string::npos) ? endpos1 : endpos2;
                sub_str = system_info_table[n][0].substr(pos, endpos1);
                core_1 = std::stoi(sub_str);
                sub_str = system_info_table[n][0].substr(endpos1 + 1);
                core_2 = std::stoi(sub_str);

                UpdateProcMapping(core_1);

                _cpu_mapping_table[core_2][CPU_MAP_PROCESSOR_ID] = core_2;
                _cpu_mapping_table[core_2][CPU_MAP_SOCKET_ID] = _cpu_mapping_table[core_1][CPU_MAP_SOCKET_ID];
                _cpu_mapping_table[core_2][CPU_MAP_CORE_ID] = _cpu_mapping_table[core_1][CPU_MAP_CORE_ID];
                _cpu_mapping_table[core_2][CPU_MAP_CORE_TYPE] = HYPER_THREADING_PROC;
                _cpu_mapping_table[core_2][CPU_MAP_GROUP_ID] = _cpu_mapping_table[core_1][CPU_MAP_GROUP_ID];
            } else if (pos != std::string::npos) {
                core_1 = std::stoi(system_info_table[n][0]);
                UpdateProcMapping(core_1);

                _cpu_mapping_table[core_1][CPU_MAP_GROUP_ID] = _cores;
            }

            _cores++;
        }
    }

    if (_sockets >= 1) {
        _proc_type_table.resize(_sockets + 2, std::vector<int>(PROC_TYPE_TABLE_SIZE, 0));
        for (int n = 0; n < _processors; n++) {
            _proc_type_table[0][ALL_PROC]++;
            _proc_type_table[_cpu_mapping_table[n][CPU_MAP_SOCKET_ID] + 1][ALL_PROC]++;

            _proc_type_table[0][_cpu_mapping_table[n][CPU_MAP_CORE_TYPE]]++;
            _proc_type_table[_cpu_mapping_table[n][CPU_MAP_SOCKET_ID] + 1][_cpu_mapping_table[n][CPU_MAP_CORE_TYPE]]++;
        }
        _sockets++;
    } else {
        _proc_type_table.resize(1, std::vector<int>(PROC_TYPE_TABLE_SIZE, 0));
        for (int n = 0; n < _processors; n++) {
            _proc_type_table[0][ALL_PROC]++;
            _proc_type_table[0][_cpu_mapping_table[n][CPU_MAP_CORE_TYPE]]++;
            _cpu_mapping_table[n][CPU_MAP_SOCKET_ID] = 0;
        }
        _sockets = 1;
    }
};

}  // namespace ov
