// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_test_util.hpp"

#include <openvino/util/file_util.hpp>

#ifdef __APPLE__
# include <mach-o/dyld.h>
#endif

#ifdef _WIN32
// #include <libloaderapi.h>
# include <windows.h>
// # include <direct.h>
// # include <stdlib.h>
#else
// # include <dlfcn.h>
# include <unistd.h>
# include <limits.h>
#endif

namespace xuejun {

std::string getExecutableDirectory_xuejun() {
    std::string path;
#ifdef _WIN32
    char buffer[MAX_PATH];
    int len = GetModuleFileNameA(NULL, buffer, MAX_PATH);
#elif defined(__APPLE__)
    Dl_info info;
    dladdr(reinterpret_cast<void*>(getExecutableDirectory), &info);
    const char * buffer = info.dli_fname;
    int len = std::strlen(buffer);
#else
    char buffer[PATH_MAX];
    int len = readlink("/proc/self/exe", buffer, PATH_MAX);
#endif
    if (len < 0) {
        throw "Can't get test executable path name";
    }
    path = std::string(buffer, len);
    return ov::util::get_directory(path);
}
}