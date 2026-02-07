// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/lib_load_unload.hpp"

#include <vector>

namespace {
#define UNLOAD_FUNC(func_name) &func_name,
    std::vector<void(*)()> unload_functions = {
    #include "openvino/unload_functions.inc"
    };
#undef UNLOAD_FUNC

void call_on_unload() {
    for (const auto& func : unload_functions) {
        func();
    }
}
} // namespace

#if defined(_WIN32) && !defined(__MINGW32__) && !defined(__MINGW64__)
#    include <windows.h>
BOOL WINAPI DllMain(HINSTANCE hinstDLL,  // handle to DLL module
                    DWORD fdwReason,     // reason for calling function
                    LPVOID lpReserved)   // reserved
{
    // Perform actions based on the reason for calling.
    switch (fdwReason) {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
        break;

    case DLL_PROCESS_DETACH:
        call_on_unload();
        break;
    }
    return TRUE;  // Successful DLL_PROCESS_ATTACH.
}
#elif defined(__linux__) || defined(__APPLE__) || defined(__EMSCRIPTEN__)
extern "C" __attribute__((destructor)) void library_unload();
void library_unload() {
    call_on_unload();
}
#endif
