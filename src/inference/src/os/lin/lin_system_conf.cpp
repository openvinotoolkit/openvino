// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sched.h>
#include <string.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "ie_common.h"
#include "ie_system_conf.h"
#include "streams_executor.hpp"
#include "threading/ie_parallel_custom_arena.hpp"

namespace InferenceEngine {

struct CPU {
    int _processors = 0;
    int _sockets = 0;
    int _cores = 0;

    std::vector<std::vector<int>> _proc_type_table;
    std::vector<std::vector<int>> _cpu_mapping_table;

    CPU() {
        std::vector<std::vector<std::string>> system_info_table;

        auto GetCatchInfoLinux = [&]() {
            _processors = sysconf(_SC_NPROCESSORS_ONLN);
            system_info_table.resize(_processors, std::vector<std::string>(3));

            for (int n = 0; n < _processors; n++) {
                for (int m = 0; m < 3; m++) {
                    int Ln = (m == 0) ? m : m + 1;

                    std::ifstream cache_file("/sys/devices/system/cpu/cpu" + std::to_string(n) + "/cache/index" +
                                             std::to_string(Ln) + "/shared_cpu_list");
                    if (!cache_file.is_open()) {
                        return -1;
                    }
                    std::string cache_info;
                    std::getline(cache_file, cache_info);
                    system_info_table[n][m] += cache_info;
                }
            }
            return 0;
        };

        if (!GetCatchInfoLinux()) {
            parse_processor_info_linux(_processors,
                                       system_info_table,
                                       _sockets,
                                       _cores,
                                       _proc_type_table,
                                       _cpu_mapping_table);
        } else {
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
            _sockets = sockets.size();
            for (auto&& socket : sockets) {
                _cores += socket.second;
            }
            if (_cores == 0) {
                _cores = _processors;
            }
        }
        std::vector<std::vector<std::string>>().swap(system_info_table);
    }
};
static CPU cpu;

void parse_processor_info_linux(const int _processors,
                                const std::vector<std::vector<std::string>> system_info_table,
                                int& _sockets,
                                int& _cores,
                                std::vector<std::vector<int>>& _proc_type_table,
                                std::vector<std::vector<int>>& _cpu_mapping_table) {
    int n_group = 0;

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

#if !((IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO))
std::vector<int> getAvailableNUMANodes() {
    std::vector<int> nodes((0 == cpu._sockets) ? 1 : cpu._sockets);
    std::iota(std::begin(nodes), std::end(nodes), 0);
    return nodes;
}
#endif
int getNumberOfCPUCores(bool bigCoresOnly) {
    unsigned numberOfProcessors = cpu._processors;
    unsigned totalNumberOfCpuCores = cpu._cores;
    IE_ASSERT(totalNumberOfCpuCores != 0);
    cpu_set_t usedCoreSet, currentCoreSet, currentCpuSet;
    CPU_ZERO(&currentCpuSet);
    CPU_ZERO(&usedCoreSet);
    CPU_ZERO(&currentCoreSet);

    sched_getaffinity(0, sizeof(currentCpuSet), &currentCpuSet);

    for (unsigned processorId = 0u; processorId < numberOfProcessors; processorId++) {
        if (CPU_ISSET(processorId, &currentCpuSet)) {
            unsigned coreId = processorId % totalNumberOfCpuCores;
            if (!CPU_ISSET(coreId, &usedCoreSet)) {
                CPU_SET(coreId, &usedCoreSet);
                CPU_SET(processorId, &currentCoreSet);
            }
        }
    }
    int phys_cores = CPU_COUNT(&currentCoreSet);
#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
    auto core_types = custom::info::core_types();
    if (bigCoresOnly && core_types.size() > 1) /*Hybrid CPU*/ {
        phys_cores = custom::info::default_concurrency(
            custom::task_arena::constraints{}.set_core_type(core_types.back()).set_max_threads_per_core(1));
    }
#endif
    return phys_cores;
}

}  // namespace InferenceEngine
