// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>

#include <openvino/runtime/intel_npu/properties.hpp>

#ifdef _linux_
#include <dlfcn.h>
#endif

#include <sstream>
#ifdef WIN32
#   include <process.h>
#   include <windows.h>
#   include <libloaderapi.h>
#endif
#include "gtest/gtest.h"

#ifdef _linux_
void* (*_dlopen)(const char *filename, int mode);
char* (*_dlerror)(void);

__attribute__((constructor))
void init() {
    _dlopen = reinterpret_cast<decltype(_dlopen)>(dlsym(RTLD_NEXT, "dlopen"));
    _dlerror = reinterpret_cast<decltype(_dlerror)>(dlsym(RTLD_NEXT, "dlerror"));
}

thread_local char* dlopen_error_str = nullptr;

extern "C" void* dlopen(const char *filename, int mode) {   
    if (filename && strstr(filename, "libze_intel_npu.so")) {
        dlopen_error_str = (char*)"Failed to dlopen libze_intel_npu.so";
        return nullptr;
    }
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
#endif

#ifdef __cplusplus
    #define INITIALIZER(f) \
        static void f(void); \
        struct f##_t_ { f##_t_(void) { f(); } }; static f##_t_ f##_; \
        static void f(void)
#elif defined(_MSC_VER)
    #pragma section(".CRT$XCU",read)
    #define INITIALIZER2_(f,p) \
        static void f(void); \
        __declspec(allocate(".CRT$XCU")) void (*f##_)(void) = f; \
        __pragma(comment(linker,"/include:" p #f "_")) \
        static void f(void)
    #ifdef _WIN64
        #define INITIALIZER(f) INITIALIZER2_(f,"")
    #else
        #define INITIALIZER(f) INITIALIZER2_(f,"_")
    #endif
#endif

#ifdef WIN32
    HMODULE (*_LoadLibraryW)(LPCWSTR lpLibFileName);

    INITIALIZER(init) {
        _LoadLibraryW = reinterpret_cast<decltype(_LoadLibraryW)>(GetProcAddress(GetModuleHandle(TEXT("kernel32.dll")), TEXT("LoadLibraryW")));
    }

    __declspec(dllimport)
    inline HMODULE LoadLibraryW(LPCWSTR lpLibFileName) {
        if (lpLibFileName && lstrcmpW(lpLibFileName, L"npu_level_zero_umd.dll") == 0) {
            return nullptr;
        }
        return _LoadLibraryW(lpLibFileName);
    }
#endif

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
