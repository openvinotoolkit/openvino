// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"

#include "gtest/gtest.h"
#include "openvino/core/except.hpp"
#include "precomp.hpp"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    define _WINSOCKAPI_

#    include <windows.h>

#    include "psapi.h"
#endif

namespace ov {
namespace test {
namespace utils {

std::ostream& operator<<(std::ostream& os, OpType type) {
    switch (type) {
    case OpType::SCALAR:
        os << "SCALAR";
        break;
    case OpType::VECTOR:
        os << "VECTOR";
        break;
    default:
        OPENVINO_THROW("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::string generateTestFilePrefix() {
    // Generate unique file names based on test name, thread id and timestamp
    // This allows execution of tests in parallel (stress mode)
    auto testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string testName = testInfo->test_case_name();
    testName += testInfo->name();
    testName = std::to_string(std::hash<std::string>()(testName));
    std::stringstream ss;
    auto ts = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch());
    ss << testName << "_" << std::this_thread::get_id() << "_" << ts.count();
    testName = ss.str();
    std::replace(testName.begin(), testName.end(), ':', '_');
    return testName;
}

#ifdef _WIN32
static PROCESS_MEMORY_COUNTERS getMemoryInfo() {
    static PROCESS_MEMORY_COUNTERS pmc;
    pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, pmc.cb))
        throw std::runtime_error("Can't get system memory values");
    return pmc;
}

size_t getVmSizeInKB() {
    return getMemoryInfo().PagefileUsage / 1024;
}

size_t getVmRSSInKB() {
    return getMemoryInfo().WorkingSetSize / 1024;
}

#else

size_t getSystemDataByName(char* name) {
    auto parseLine = [](std::string line) -> size_t {
        std::string res = "";
        for (auto c : line)
            if (isdigit(c))
                res += c;
        if (res.empty())
            throw std::runtime_error("Can't get system memory values");
        return std::stoul(res);
    };

    FILE* file = fopen("/proc/self/status", "r");
    size_t result = 0;
    bool status = false;
    if (file != nullptr) {
        char line[128];

        while (fgets(line, 128, file) != NULL) {
            if (strncmp(line, name, strlen(name)) == 0) {
                result = parseLine(line);
                status = true;
                break;
            }
        }
        fclose(file);
    }
    if (!status)
        throw std::runtime_error("Can't get system memory values");
    return result;
}

size_t getVmSizeInKB() {
    return getSystemDataByName(const_cast<char*>("VmSize:"));
}

size_t getVmRSSInKB() {
    return getSystemDataByName(const_cast<char*>("VmRSS:"));
}

#endif
}  // namespace utils
}  // namespace test
}  // namespace ov
