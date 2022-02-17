// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_common.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    define _WINSOCKAPI_

#    include <windows.h>

#    include "Psapi.h"
#endif

namespace ov {
namespace test {

inline size_t getVmSizeInKB() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS);
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, pmc.cb);
    return pmc.WorkingSetSize;
#else
    auto parseLine = [](char* line) {
        // This assumes that a digit will be found and the line ends in " Kb".
        size_t i = strlen(line);
        const char* p = line;
        while (*p < '0' || *p > '9')
            p++;
        line[i - 3] = '\0';
        i = (size_t)atoi(p);
        return i;
    };

    FILE* file = fopen("/proc/self/status", "r");
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

TestsCommon::~TestsCommon() = default;

TestsCommon::TestsCommon() {
    auto memsize = getVmSizeInKB();
    if (memsize != 0) {
        std::cout << "\nMEM_USAGE=" << memsize << "KB\n";
    }
}

std::string TestsCommon::GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto epoch = now.time_since_epoch();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch);
    return std::to_string(ns.count());
}

std::string TestsCommon::GetTestName() const {
    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::replace_if(
        test_name.begin(),
        test_name.end(),
        [](char c) {
            return !std::isalnum(c);
        },
        '_');
    return test_name;
}

}  // namespace test
std::shared_ptr<SharedRTInfo> ModelAccessor::get_shared_info() const {
    if (auto f = m_function.lock()) {
        return f->m_shared_rt_info;
    }
    throw ngraph::ngraph_error("Original model is not available");
}

std::set<std::shared_ptr<SharedRTInfo>> NodeAccessor::get_shared_info() const {
    if (auto node = m_node.lock()) {
        return node->m_shared_rt_info;
    }
    throw ngraph::ngraph_error("Original node is not available");
}
}  // namespace ov
