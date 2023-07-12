// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_common.hpp"

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "precomp.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define _WINSOCKAPI_

#include <windows.h>
#include "psapi.h"
#endif

#ifdef ENABLE_CONFORMANCE_PGQL
#    include "common_test_utils/postgres_link.hpp"
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
    auto parseLine = [](char *line) {
        // This assumes that a digit will be found and the line ends in " Kb".
        size_t i = strlen(line);
        const char *p = line;
        while (*p < '0' || *p > '9')
            p++;
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
    ov::threading::executor_manager()->clear();

#ifdef ENABLE_CONFORMANCE_PGQL
    delete PGLink;
    PGLink = nullptr;
#endif
}

TestsCommon::TestsCommon()
#ifdef ENABLE_CONFORMANCE_PGQL
    : PGLink(new PostgreSQLLink(this))
#endif
{
    auto memsize = getVmSizeInKB();
    if (memsize != 0) {
        std::cout << "\nMEM_USAGE=" << memsize << "KB\n";
    }
    ov::threading::executor_manager()->clear();
}

std::string TestsCommon::GetTimestamp() {
    return ov::test::utils::GetTimestamp();
}

std::string TestsCommon::GetTestName() const {
    std::string test_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::replace_if(test_name.begin(), test_name.end(),
        [](char c) { return !std::isalnum(c); }, '_');
    return test_name;
}

std::string TestsCommon::GetFullTestName() const {
    std::string suite_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name();
    std::replace_if(suite_name.begin(), suite_name.end(),
        [](char c) { return !std::isalnum(c); }, '_');

    std::string test_name = GetTestName();

    return suite_name + "_" + test_name;
}

}  // namespace test
std::shared_ptr<SharedRTInfo> ModelAccessor::get_shared_info() const {
    if (auto f = m_function.lock()) {
        return f->m_shared_rt_info;
    }
    OPENVINO_THROW("Original model is not available");
}

std::set<std::shared_ptr<SharedRTInfo>> NodeAccessor::get_shared_info() const {
    if (auto node = m_node.lock()) {
        return node->m_shared_rt_info;
    }
    OPENVINO_THROW("Original node is not available");
}
}  // namespace ov