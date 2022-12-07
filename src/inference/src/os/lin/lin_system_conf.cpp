// Copyright (C) 2018-2022 Intel Corporation
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
         * New method to get CPU infomation and CPU map
         */
        _proc = sysconf(_SC_NPROCESSORS_ONLN);
        int _cpu_mapping[_proc][CPU_MAP_USED_PROC + 1];
        memset(_cpu_mapping, 0, _proc * (CPU_MAP_USED_PROC + 1) * sizeof(int));

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

}  // namespace InferenceEngine
