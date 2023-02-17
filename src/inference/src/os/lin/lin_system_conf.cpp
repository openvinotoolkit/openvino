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

std::vector<std::vector<int>> getNumOfAvailableCPUCores() {
    std::vector<std::vector<int>> proc_type_table;
    std::vector<int> all_table;
    if (cpu._sockets == 1) {
        proc_type_table.resize(1, std::vector<int>(PROC_TYPE_TABLE_SIZE, 0));
    } else {
        proc_type_table.resize(2, std::vector<int>(PROC_TYPE_TABLE_SIZE, 0));
    }
    all_table.resize(PROC_TYPE_TABLE_SIZE, 0);
    for (int i = 0; i < cpu._processors; i++) {
        for (int socket_id = 0; socket_id < cpu._sockets; socket_id++) {
            for (int type = MAIN_CORE_PROC; type < PROC_TYPE_TABLE_SIZE; type++) {
                if (cpu._cpu_mapping_table[i][CPU_MAP_CORE_TYPE] == type &&
                    cpu._cpu_mapping_table[i][CPU_MAP_SOCKET_ID] == socket_id &&
                    cpu._cpu_mapping_table[i][CPU_MAP_USED_FLAG] == -1) {
                    proc_type_table[socket_id][type]++;
                    proc_type_table[socket_id][ALL_PROC]++;
                    all_table[type]++;
                    all_table[ALL_PROC]++;
                }
            }
        }
    }
    if (cpu._sockets > 1) {
        proc_type_table.insert(proc_type_table.begin(), all_table);
    }
    return proc_type_table;
}

bool cpuMapAvailable() {
    return cpu._cpu_mapping_table.size() > 0;
}

std::vector<int> getAvailableCPUs(const column_of_processor_type_table core_type, const int num_cpus, const bool cpu_task) {
    std::vector<int> cpu_ids;
    const int used_flag = cpu_task ? -1 : 2;
    if (core_type < PROC_TYPE_TABLE_SIZE && core_type >= ALL_PROC) {
        for (int i = 0; i < cpu._processors; i++) {
            if (cpu._cpu_mapping_table[i][CPU_MAP_CORE_TYPE] == core_type &&
                cpu._cpu_mapping_table[i][CPU_MAP_USED_FLAG] == used_flag) {
                cpu_ids.push_back(cpu._cpu_mapping_table[i][CPU_MAP_PROCESSOR_ID]);
            }
            if (static_cast<int>(cpu_ids.size()) == num_cpus) {
                break;
            }
        }
    } else {
        IE_THROW() << "Wrong value for core_type " << core_type;
    }
    return cpu_ids;
}

std::vector<int> getLogicCores(std::vector<int> cpu_ids) {
    std::vector<int> logic_cores;
    int cpu_size = static_cast<int>(cpu_ids.size());
    for (int i = 0; i < cpu._processors; i++) {
        for (int j = 0; j < cpu_size; j++) {
            if (cpu._cpu_mapping_table[i][CPU_MAP_CORE_ID] == cpu._cpu_mapping_table[cpu_ids[j]][CPU_MAP_CORE_ID] &&
                cpu._cpu_mapping_table[i][CPU_MAP_PROCESSOR_ID] != cpu_ids[j]) {
                logic_cores.push_back(cpu._cpu_mapping_table[i][CPU_MAP_PROCESSOR_ID]);
            }
        }
        if (cpu_ids.size() == logic_cores.size()) {
            break;
        }
    }
    return logic_cores;
}

void setCpuUsed(std::vector<int> cpu_ids, int used) {
    const auto cpu_size = static_cast<int>(cpu_ids.size());
    for (int i = 0; i < cpu_size; i++) {
        if (cpu_ids[i] < cpu._processors) {
            cpu._cpu_mapping_table[cpu_ids[i]][CPU_MAP_USED_FLAG] = used;
        }
    }
}

}  // namespace InferenceEngine
