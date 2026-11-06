// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <level_zero/ze_api.h>

#include "openvino/core/except.hpp"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

// The Level Zero loader ships with the GPU driver, not with OpenVINO. Like the GPU plugin, the sample loads it
// at runtime instead of linking it, so it starts on machines without the driver and reports a clear error.
// Usage: ze::zeInitDrivers(&count, nullptr, &desc) instead of zeInitDrivers(&count, nullptr, &desc).
namespace ze {

// clang-format off
#define LEVEL_ZERO_SYMBOLS(symbol)                      \
    symbol(zeInitDrivers)                               \
    symbol(zeDeviceGetCommandQueueGroupProperties)      \
    symbol(zeCommandListCreateImmediate)                \
    symbol(zeCommandListDestroy)                        \
    symbol(zeCommandListAppendMemoryCopy)               \
    symbol(zeCommandListHostSynchronize)                \
    symbol(zeMemGetAllocProperties)
// clang-format on

class Loader {
public:
    static const Loader& get() {
        // Loaded on first use, i.e. after the command line and the device name were validated.
        // Never unloaded: the driver may still be in use by OpenVINO at exit.
        static const Loader loader;
        return loader;
    }

    Loader(const Loader&) = delete;
    Loader& operator=(const Loader&) = delete;

#define DECLARE_SYMBOL(symbol) decltype(&::symbol) symbol = nullptr;
    LEVEL_ZERO_SYMBOLS(DECLARE_SYMBOL)
#undef DECLARE_SYMBOL

private:
    Loader() {
#ifdef _WIN32
        const char* library = "ze_loader.dll";
        _handle = LoadLibraryA(library);
#else
        const char* library = "libze_loader.so.1";
        _handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
#endif
        OPENVINO_ASSERT(_handle != nullptr,
                        "Cannot load the Level Zero loader ",
                        library,
                        ". Is the GPU driver installed?");
#define RESOLVE_SYMBOL(symbol) resolve(symbol, #symbol);
        LEVEL_ZERO_SYMBOLS(RESOLVE_SYMBOL)
#undef RESOLVE_SYMBOL
    }

    template <typename Function>
    void resolve(Function& function, const char* name) {
#ifdef _WIN32
        function = reinterpret_cast<Function>(GetProcAddress(_handle, name));
#else
        function = reinterpret_cast<Function>(dlsym(_handle, name));
#endif
        OPENVINO_ASSERT(function != nullptr, "The Level Zero loader has no ", name, ". Update the GPU driver.");
    }

#ifdef _WIN32
    HMODULE _handle = nullptr;
#else
    void* _handle = nullptr;
#endif
};

// ze::zeXxx(...) forwards to the function resolved from the loader.
#define DEFINE_SYMBOL(symbol)                 \
    template <typename... Args>               \
    ze_result_t symbol(Args... args) {        \
        return Loader::get().symbol(args...); \
    }
LEVEL_ZERO_SYMBOLS(DEFINE_SYMBOL)
#undef DEFINE_SYMBOL

#undef LEVEL_ZERO_SYMBOLS

}  // namespace ze
