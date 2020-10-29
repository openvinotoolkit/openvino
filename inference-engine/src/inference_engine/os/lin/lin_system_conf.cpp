// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <sched.h>
#include "ie_system_conf.h"
#include "ie_parallel.hpp"
#include "details/ie_exception.hpp"
#include <numeric>


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
int getNumberOfCPUCores() {
    unsigned numberOfProcessors = cpu._processors;
    unsigned totalNumberOfCpuCores = cpu._cores;
    IE_ASSERT(totalNumberOfCpuCores != 0);
    cpu_set_t usedCoreSet, currentCoreSet, currentCpuSet;
    CPU_ZERO(&currentCpuSet);
    CPU_ZERO(&usedCoreSet);
    CPU_ZERO(&currentCoreSet);

    sched_getaffinity(0, sizeof(currentCpuSet), &currentCpuSet);

    for (int processorId = 0; processorId < numberOfProcessors; processorId++) {
        if (CPU_ISSET(processorId, &currentCpuSet)) {
            unsigned coreId = processorId % totalNumberOfCpuCores;
            if (!CPU_ISSET(coreId, &usedCoreSet)) {
                CPU_SET(coreId, &usedCoreSet);
                CPU_SET(processorId, &currentCoreSet);
            }
        }
    }
    return CPU_COUNT(&currentCoreSet);
}

}  // namespace InferenceEngine
