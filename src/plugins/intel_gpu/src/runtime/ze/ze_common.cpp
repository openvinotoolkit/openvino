// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_common.hpp"
#include "openvino/core/except.hpp"

#if defined(__linux__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include "windows.h"
#else
#error "Level Zero is supported on Linux and Windows only"
#endif

namespace cldnn {
namespace ze {

void *find_ze_symbol(const char *symbol) {
#if defined(__linux__)
    void *handle = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
#elif defined(_WIN32)
    HMODULE handle = LoadLibraryA("ze_loader.dll");
#endif
    if (!handle) {
        return nullptr;
    }

#if defined(__linux__)
    void *f = dlsym(handle, symbol);
#elif defined(_WIN32)
    void *f = GetProcAddress(handle, symbol);
#endif
    OPENVINO_ASSERT(f != nullptr);
    return f;
}

}  // namespace ze
}  // namespace cldnn
