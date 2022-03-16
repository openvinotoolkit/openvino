// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_common.hpp"
#include "common_utils.hpp"

#include <threading/ie_executor_manager.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <random>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define _WINSOCKAPI_

#include <windows.h>
#include "Psapi.h"
#endif

namespace CommonTestUtils {

inline std::vector<size_t> getVmSizeInKB() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
        pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS);
        GetProcessMemoryInfo(GetCurrentProcess(), &pmc, pmc.cb);
        return {pmc.WorkingSetSize, 0, 0};
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
    size_t vmSize = 0;
    size_t rssSize = 0;
    size_t rssMaxSize = 0;
    if (file != nullptr) {
        char line[128];

        while (fgets(line, 128, file) != NULL) {
            if (strncmp(line, "VmSize:", 7) == 0) {
                vmSize = parseLine(line);
            }
            if (strncmp(line, "VmHWM:", 6) == 0) {
                rssMaxSize = parseLine(line);
            }
            if (strncmp(line, "VmRSS:", 6) == 0) {
                rssSize = parseLine(line);
                break;
            }
        }
        fclose(file);
    }
    return {vmSize, rssSize, rssMaxSize};
#endif
}

TestsCommon::~TestsCommon() {
    InferenceEngine::executorManager()->clear();
}

TestsCommon::TestsCommon() {
    auto memsize = getVmSizeInKB();
    std::cout << "\n";
    if (memsize.at(0) != 0) {
        std::cout << "VM MEM_USAGE=" << memsize[0] << "KB\n";
    }
    if (memsize.at(1) != 0) {
        std::cout << "Rss MEM_USAGE=" << memsize[1] << "KB\n";
    }
    if (memsize.at(2) != 0) {
        std::cout << "Rss Max MEM_USAGE=" << memsize[2] << "KB\n";
    }
    std::cout << "\n";

    InferenceEngine::executorManager()->clear();
}

std::string TestsCommon::GetTimestamp() {
    return CommonTestUtils::GetTimestamp();
}

std::string TestsCommon::GetTestName() const {
    std::string test_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::replace_if(test_name.begin(), test_name.end(),
        [](char c) { return !std::isalnum(c); }, '_');
    return test_name;
}

}  // namespace CommonTestUtils
