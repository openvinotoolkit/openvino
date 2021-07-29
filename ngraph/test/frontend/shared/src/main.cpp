// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>

#include <ngraph/file_util.hpp>
#include "gtest/gtest.h"

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#ifdef _WIN32
#include <Windows.h>
#else
#include <limits.h>
#endif

using namespace std;

namespace FrontEndTestUtils
{
    int run_tests(int argc, char** argv)
    {
        ::testing::InitGoogleTest(&argc, argv);
        int rc = RUN_ALL_TESTS();
        return rc;
    }

    std::string get_current_executable_path()
    {
        std::string path;
#ifdef _WIN32
        char buffer[MAX_PATH];
        int len = GetModuleFileNameA(NULL, buffer, MAX_PATH);
#elif defined(__APPLE__)
        char buffer[PATH_MAX];
        uint32_t len = 1024; // Definitely enough for test purposes
        if (_NSGetExecutablePath(buffer, &len) != 0)
        {
            throw "Can't get test executable path name";
        }
#else
        char buffer[PATH_MAX];
        int len = readlink("/proc/self/exe", buffer, PATH_MAX);
#endif
        if (len < 0)
        {
            throw "Can't get test executable path name";
        }
        path = std::string(buffer, len);
        return ngraph::file_util::get_directory(path);
    }
} // namespace FrontEndTestUtils