// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shutdown.hpp"

// Shutdown functions section declaration
#if defined(_MSC_VER)
// Get the release start/end addresses (MSVC)
#    pragma section(".shutdown_sec$a", read)
#    pragma section(".shutdown_sec$z", read)
extern "C" __declspec(allocate(".shutdown_sec$a")) void (*__ov_shutdown_start)(void) = nullptr;
extern "C" __declspec(allocate(".shutdown_sec$z")) void (*__ov_shutdown_end)(void) = nullptr;

#elif (defined(__GNUC__) || defined(__clang__))

#    if defined(__APPLE__)
extern "C" void (*__ov_shutdown_start)(void) __asm("section$start$__DATA$__shutdown_sec");
extern "C" void (*__ov_shutdown_end)(void) __asm("section$end$__DATA$__shutdown_sec");
#    else
// Get the release start/end addresses (GCC/Clang)
extern "C" void (*__ov_shutdown_start)(void);
extern "C" void (*__ov_shutdown_end)(void);
#    endif

#else
#    error "Compiler not supported"
#endif

static void shutdown_resources() {
    void (**start)(void) = &__ov_shutdown_start;
    void (**end)(void) = &__ov_shutdown_end;

    for (void (**func)(void) = start; func != end; ++func) {
        if (*func) {
            (**func)();
        }
    }
}

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
