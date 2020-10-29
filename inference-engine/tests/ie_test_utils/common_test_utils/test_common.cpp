// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_common.hpp"

#include <threading/ie_executor_manager.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define _WINSOCKAPI_

#include <windows.h>
#include "Psapi.h"
#endif

namespace CommonTestUtils {

inline size_t getVmSizeInKB() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
        pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS);
        GetProcessMemoryInfo(GetCurrentProcess(), &pmc, pmc.cb);
        return pmc.WorkingSetSize;
#else
    auto parseLine = [](char *line) {
        // This assumes that a digit will be found and the line ends in " Kb".
        size_t i = strlen(line);
        const char *p = line;
        while (*p < '0' || *p > '9') p++;
        line[i - 3] = '\0';
        i = (size_t) atoi(p);
        return i;
    };

    FILE *file = fopen("/proc/self/status", "r");
    size_t result = 0;
    if (file != nullptr) {
        char line[128];

        while (fgets(line, 128, file) != NULL) {
            if (strncmp(line, "VmSize:", 7) == 0) {
                result = parseLine(line);
                break;
            }
        }
        fclose(file);
    }
    return result;
#endif
}

TestsCommon::~TestsCommon() {
    InferenceEngine::ExecutorManager::getInstance()->clear();
}

TestsCommon::TestsCommon() {
    auto memsize = getVmSizeInKB();
    if (memsize != 0) {
        std::cout << "\nMEM_USAGE=" << memsize << "KB\n";
    }
    InferenceEngine::ExecutorManager::getInstance()->clear();
}

}  // namespace CommonTestUtils