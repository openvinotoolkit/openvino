// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>

#include <openvino/runtime/intel_npu/properties.hpp>

#ifdef __linux__
#    include <dlfcn.h>
#endif

#include <sstream>
#ifdef WIN32
// clang-format off
#    include <process.h>
#    include <windows.h>
#    include <libloaderapi.h>
// clang-format on
#endif
#include "gtest/gtest.h"

#ifdef __linux__
void* (*_dlopen)(const char* filename, int mode);
char* (*_dlerror)(void);

__attribute__((constructor)) static void init() {
    _dlopen = reinterpret_cast<decltype(_dlopen)>(dlsym(RTLD_NEXT, "dlopen"));
    _dlerror = reinterpret_cast<decltype(_dlerror)>(dlsym(RTLD_NEXT, "dlerror"));
}

thread_local char* dlopen_error_str = nullptr;

extern "C" __attribute__((visibility("default"))) void* dlopen(const char* filename, int mode) {
    if (!_dlopen) {
        _dlopen = reinterpret_cast<decltype(_dlopen)>(dlsym(RTLD_NEXT, "dlopen"));
    }
    if (filename && strstr(filename, "libze_intel_npu.so")) {
        dlopen_error_str = (char*)"Failed to dlopen libze_intel_npu.so";
        return nullptr;
    }
    return _dlopen(filename, mode);
}

extern "C" __attribute__((visibility("default"))) char* dlerror(void) {
    if (!_dlerror) {
        _dlerror = reinterpret_cast<decltype(_dlerror)>(dlsym(RTLD_NEXT, "dlerror"));
    }
    if (dlopen_error_str) {
        char* ret = dlopen_error_str;
        dlopen_error_str = nullptr;
        return ret;
    }
    return _dlerror();
}
#endif

#ifdef __cplusplus
#    define INITIALIZER(f)   \
        static void f(void); \
        struct f##_t_ {      \
            f##_t_(void) {   \
                f();         \
            }                \
        };                   \
        static f##_t_ f##_;  \
        static void f(void)
#elif defined(_MSC_VER)
#    pragma section(".CRT$XCU", read)
#    define INITIALIZER2_(f, p)                                  \
        static void f(void);                                     \
        __declspec(allocate(".CRT$XCU")) void (*f##_)(void) = f; \
        __pragma(comment(linker, "/include:" p #f "_")) static void f(void)
#    ifdef _WIN64
#        define INITIALIZER(f) INITIALIZER2_(f, "")
#    else
#        define INITIALIZER(f) INITIALIZER2_(f, "_")
#    endif
#endif

#if defined _WIN32 || defined _WIN64
/*
Inline hook for LoadLibraryW — intercepts ALL callers in the process,
including DLLs, exactly like the Linux dlopen override above.

Strategy:
  Overwrite the entry of KernelBase!LoadLibraryW with a 14-byte absolute
  indirect JMP to our hook.  For the passthrough we call LoadLibraryExW
  directly (LoadLibraryW is just LoadLibraryExW(name, NULL, 0)), which
  avoids any trampoline that would need a disassembler to safely relocate
  RIP-relative instructions.
*/

typedef HMODULE(WINAPI* LoadLibraryExW_t)(LPCWSTR, HANDLE, DWORD);
static LoadLibraryExW_t g_LoadLibraryExW = nullptr;

static const size_t HOOK_SIZE = 14;  // FF 25 00000000 <addr64>

static HMODULE WINAPI Hooked_LoadLibraryW(LPCWSTR lpLibFileName) {
    if (lpLibFileName && wcsstr(lpLibFileName, L"ze_loader.dll") != nullptr) {
        SetLastError(ERROR_MOD_NOT_FOUND);
        return nullptr;
    }
    // LoadLibraryW(name) == LoadLibraryExW(name, NULL, 0) — call the real
    // implementation directly, bypassing our patched LoadLibraryW entry.
    return g_LoadLibraryExW(lpLibFileName, nullptr, 0);
}

// Write a 14-byte absolute indirect JMP: FF 25 00000000 <addr64>
static void WriteAbsJmp64(BYTE* dst, void* target) {
    dst[0] = 0xFF;
    dst[1] = 0x25;
    *reinterpret_cast<DWORD*>(dst + 2) = 0;  // JMP [RIP+0]
    *reinterpret_cast<ULONG_PTR*>(dst + 6) = reinterpret_cast<ULONG_PTR>(target);
}

INITIALIZER(init) {
    HMODULE hKernelBase = GetModuleHandleA("KernelBase.dll");
    HMODULE hMod = hKernelBase ? hKernelBase : GetModuleHandleA("kernel32.dll");

    g_LoadLibraryExW = reinterpret_cast<LoadLibraryExW_t>(GetProcAddress(hMod, "LoadLibraryExW"));
    BYTE* real = reinterpret_cast<BYTE*>(GetProcAddress(hMod, "LoadLibraryW"));
    if (!real || !g_LoadLibraryExW)
        return;

    DWORD oldProtect = 0;
    VirtualProtect(real, HOOK_SIZE, PAGE_EXECUTE_READWRITE, &oldProtect);
    WriteAbsJmp64(real, reinterpret_cast<void*>(Hooked_LoadLibraryW));
    VirtualProtect(real, HOOK_SIZE, oldProtect, &oldProtect);
    FlushInstructionCache(GetCurrentProcess(), real, HOOK_SIZE);
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
