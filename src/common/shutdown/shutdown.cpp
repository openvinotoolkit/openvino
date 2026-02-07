// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shutdown.hpp"

void shutdown_resources();

#if defined(SHUTDOWN_STATIC_LIBRARY)
// Shutdown callback registration mechanism
// ...existing code...
#include <functional>
#include <vector>

namespace ov {
static std::vector<std::function<void()>> g_shutdown_callbacks;

bool register_shutdown_callback(const std::function<void()>& func) {
    g_shutdown_callbacks.emplace_back(func);
    return true;
}

const std::vector<std::function<void()>>& shutdown_callbacks() {
    return g_shutdown_callbacks;
}
}  // namespace ov

void shutdown_resources() {
    for (auto& func : ov::shutdown_callbacks()) {
        if (func) {
            func();
        }
    }
}

#else

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
        shutdown_resources();
        break;
    }
    return TRUE;  // Successful DLL_PROCESS_ATTACH.
}
#elif defined(__linux__) || defined(__APPLE__) || defined(__EMSCRIPTEN__)
extern "C" __attribute__((destructor)) void library_unload();
void library_unload() {
    shutdown_resources();
}
#endif

#endif