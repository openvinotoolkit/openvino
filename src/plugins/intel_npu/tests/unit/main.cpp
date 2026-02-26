// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>
#include <dlfcn.h>

#include <sstream>
#ifdef WIN32
#    include <process.h>
#endif
#include "gtest/gtest.h"


void* (*_dlopen)(const char *filename, int mode);
char* (*_dlerror)(void);

__attribute__((constructor))
void init() {
    _dlopen = reinterpret_cast<decltype(_dlopen)>(dlsym(RTLD_NEXT, "dlopen"));
    _dlerror = reinterpret_cast<decltype(_dlerror)>(dlsym(RTLD_NEXT, "dlerror"));
}

thread_local char* dlopen_error_str = nullptr;

extern "C" void* dlopen(const char *filename, int mode) {
#define TRICK 1
#if TRICK
    if (filename && strstr(filename, "libze_intel_npu.so")) {
        dlopen_error_str = (char*)"Failed to dlopen libze_intel_npu.so";
        return nullptr;
    }
#endif
    return _dlopen(filename, mode);
}

extern "C" char* dlerror(void) {
    if (dlopen_error_str) {
        char* ret = dlopen_error_str;
        dlopen_error_str = nullptr;
        return ret;
    }
    return _dlerror();
}

void sigsegv_handler(int errCode);

void sigsegv_handler(int errCode) {
    std::cerr << "Unexpected application crash with code: " << errCode << std::endl;
    std::abort();
}

int main(int argc, char** argv, char** envp) {
    // register crashHandler for SIGSEGV signal
    signal(SIGSEGV, sigsegv_handler);

    std::ostringstream oss;
    oss << "Command line args (" << argc << "): ";
    for (int c = 0; c < argc; ++c) {
        oss << " " << argv[c];
    }
    oss << std::endl;

#ifdef WIN32
    oss << "Process id: " << _getpid() << std::endl;
#else
    oss << "Process id: " << getpid() << std::endl;
#endif

    std::cout << oss.str();
    oss.str("");

    oss << "Environment variables: ";
    for (char** env = envp; *env != 0; env++) {
        oss << *env << "; ";
    }

    std::cout << oss.str() << std::endl;

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new testing::Environment());

    std::string dTest = ::testing::internal::GTEST_FLAG(internal_run_death_test);
    if (!dTest.empty()) {
        std::cout << "gtest death test process is running" << std::endl;
    }

    return RUN_ALL_TESTS();
}
