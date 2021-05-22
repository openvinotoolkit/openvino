// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <sched.h>

#include "ie_common.h"
#include "ie_system_conf.h"
#include "threading/ie_parallel_custom_arena.hpp"


namespace InferenceEngine {

struct CPU {
    int _processors = 0;
    int _sockets    = 0;
    int _cores      = 0;

    CPU() {
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::vector<int>    processors;
        std::map<int, int>  sockets;
        int socketId = 0;
        while (!cpuinfo.eof()) {
            std::string line;
            std::getline(cpuinfo, line);
            if (line.empty()) continue;
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
        phys_cores = custom::info::default_concurrency(custom::task_arena::constraints{}
                                                               .set_core_type(core_types.back())
                                                               .set_max_threads_per_core(1));
    }
    #endif
    return phys_cores;
}

}  // namespace InferenceEngine
