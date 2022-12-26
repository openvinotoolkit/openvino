// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sched.h>

#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "ie_common.h"
#include "ie_system_conf.h"
#include "threading/ie_parallel_custom_arena.hpp"

namespace InferenceEngine {

struct CPU {
    int _processors = 0;
    int _sockets = 0;
    int _cores = 0;

    int _node = 0;
    int _p_cores = 0;
    int _e_cores = 0;
    int _phy_cores = 0;
    int _proc = 0;
    std::vector<std::vector<int>> _cpu_mapping;

    CPU() {
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

        /**
         * New method to get CPU infomation and CPU map. Below is the structure of CPU map and two sample.
         *  1. Four processors of two Pcore
         *  2. Four processors of four Ecores shared L2 cache
         *
         *  Proc ID | Socket ID | HW Core ID | Phy Core of Pcores | Logic Core of Pcores | ID of Ecore Group | Used
         *     0         1            1                0                      1                   0             0
         *     1         1            1                1                      0                   0             0
         *     2         1            2                0                      2                   0             0
         *     3         1            2                2                      0                   0             0
         *     4         1            3                0                      0                   1             0
         *     5         1            4                0                      0                   1             0
         *     6         1            5                0                      0                   1             0
         *     7         1            6                0                      0                   1             0
         */
        _proc = sysconf(_SC_NPROCESSORS_ONLN);
        _cpu_mapping.resize(_proc);
        for (int i = 0; i < _proc; i++) {
            _cpu_mapping[i].resize(CPU_MAP_USED_PROC + 1, 0);
        }
        
        // _cpu_mapping[_proc][CPU_MAP_USED_PROC + 1] = {0};
        // memset(_cpu_mapping, 0, _proc * (CPU_MAP_USED_PROC + 1) * sizeof(int));

        auto updateProcMapping = [&](const int nproc) {
            if (0 == _cpu_mapping[nproc][CPU_MAP_CORE]) {
                int core_1 = 0;
                int core_2 = 0;
                std::string::size_type pos = 0;
                std::string::size_type endpos = 0;
                std::string sub_str = "";

                std::ifstream cache_1_file("/sys/devices/system/cpu/cpu" + std::to_string(nproc) +
                                           "/cache/index0/shared_cpu_list");
                std::ifstream cache_2_file("/sys/devices/system/cpu/cpu" + std::to_string(nproc) +
                                           "/cache/index2/shared_cpu_list");
                std::string cache_1_info;
                std::getline(cache_1_file, cache_1_info);
                std::string cache_2_info;
                std::getline(cache_2_file, cache_2_info);

                if (((endpos = cache_1_info.find(',', pos)) != std::string::npos) ||
                    ((endpos = cache_1_info.find('-', pos)) != std::string::npos)) {
                    sub_str = cache_1_info.substr(pos, endpos);
                    core_1 = std::stoi(sub_str);
                    sub_str = cache_1_info.substr(endpos + 1);
                    core_2 = std::stoi(sub_str);

                    _phy_cores++;
                    _p_cores++;
                    _cpu_mapping[core_1][CPU_MAP_CORE] = _phy_cores;
                    _cpu_mapping[core_2][CPU_MAP_CORE] = _phy_cores;

                    /**
                     * Processor 0 need to handle system interception on Linux. So use second processor as physical core
                     * and first processor as logic core
                     */
                    _cpu_mapping[core_1][CPU_MAP_LOG_CORE] = _p_cores;
                    _cpu_mapping[core_2][CPU_MAP_PHY_CORE] = _p_cores;

                } else if ((endpos = cache_2_info.find('-', pos)) != std::string::npos) {
                    sub_str = cache_2_info.substr(pos, endpos);
                    core_1 = std::stoi(sub_str);
                    sub_str = cache_2_info.substr(endpos + 1);
                    core_2 = std::stoi(sub_str);

                    _e_cores++;
                    for (int m = core_1; m <= core_2; m++) {
                        _phy_cores++;
                        _cpu_mapping[m][CPU_MAP_CORE] = _phy_cores;
                        _cpu_mapping[m][CPU_MAP_SMALL_CORE] = _e_cores;
                    }

                } else {
                    core_1 = std::stoi(cache_1_info);

                    _p_cores++;
                    _phy_cores++;
                    _cpu_mapping[core_1][CPU_MAP_CORE] = _phy_cores;
                    _cpu_mapping[core_2][CPU_MAP_PHY_CORE] = _p_cores;
                }
            }
            return;
        };

        for (int n = 0; n < _proc; n++) {
            if (0 == _cpu_mapping[n][CPU_MAP_SOCKET]) {
                std::ifstream cache_3_file("/sys/devices/system/cpu/cpu" + std::to_string(n) +
                                           "/cache/index3/shared_cpu_list");
                std::string cache_3_info;
                std::getline(cache_3_file, cache_3_info);

                std::string::size_type pos = 0;
                std::string::size_type endpos = 0;
                std::string sub_str;

                int core_1;
                int core_2;

                _node++;
                while (1) {
                    if ((endpos = cache_3_info.find('-', pos)) != std::string::npos) {
                        sub_str = cache_3_info.substr(pos, endpos);
                        core_1 = std::stoi(sub_str);
                        sub_str = cache_3_info.substr(endpos + 1);
                        core_2 = std::stoi(sub_str);

                        for (int m = core_1; m <= core_2; m++) {
                            _cpu_mapping[m][CPU_MAP_SOCKET] = _node;
                            updateProcMapping(m);
                        }

                    } else if (pos != std::string::npos) {
                        sub_str = cache_3_info.substr(pos);
                        core_1 = std::stoi(sub_str);
                        _cpu_mapping[core_1][CPU_MAP_SOCKET] = _node;
                        updateProcMapping(core_1);
                        endpos = pos;
                    }

                    if ((pos = cache_3_info.find(',', endpos)) != std::string::npos) {
                        pos++;
                    } else {
                        break;
                    }
                }
            }
        }
        for (int i = 0; i < _proc; i++) {
            std::cout << i << ": ";
            for (int j = 0; j < CPU_MAP_USED_PROC + 1; j++) {
                std::cout << _cpu_mapping[i][j] << " ";
            }
            std::cout << "\n";
        }
        /**********************/
    }
};
static CPU cpu;
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

bool cpuMapAvailable() {
    return cpu._cpu_mapping.size() > 0;
}

int getCoreOffset(const int cpu_col) {
    int offset = 0;
    if (cpu_col < CPU_MAP_USED_PROC && cpu_col >= CPU_MAP_SOCKET) {
        for (int i = 0; i < cpu._processors; i++) {
            if (cpu._cpu_mapping[i][cpu_col] > 0 && cpu._cpu_mapping[i][CPU_MAP_USED_PROC] == 0) {
                offset = i;
                break;
            }
        }
    } else {
        IE_THROW() << "Wrong value for cpu_col " << cpu_col;
    }
    return offset;
}

int getThreadStep(const int cpu_col) {
    std::vector<int> proc_array;
    if (cpu_col < CPU_MAP_USED_PROC && cpu_col >= CPU_MAP_SOCKET) {
        for (int i = 0; i < cpu._processors; i++) {
            if (cpu._cpu_mapping[i][cpu_col] > 0 && cpu._cpu_mapping[i][CPU_MAP_USED_PROC] == 0) {
                proc_array.push_back(i);
            }
            if(proc_array.size() == 2){
                break;
            }
        }
    } else {
        IE_THROW() << "Wrong value for cpu_col " << cpu_col;
    }
    return proc_array.size() == 2 ? proc_array[1] - proc_array[0] : 1;
}

}  // namespace InferenceEngine
