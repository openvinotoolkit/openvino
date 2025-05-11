// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/memory.hpp"

#if defined(_WIN32)
#include <Windows.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/param.h>
#endif

namespace ov {
namespace util {

size_t get_ram_size() {
    size_t ram_mem_size_bytes = 0;
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    ram_mem_size_bytes = status.ullTotalPhys;
#elif defined(CTL_HW) && defined(HW_MEMSIZE)
    int mib[2];
    mib[0] = CTL_HW;
#if defined(HW_MEMSIZE)
    mib[1] = HW_MEMSIZE;
#endif
    int64_t size = 0;
    size_t len = sizeof(size);
    if (sysctl(mib, 2, &size, &len, NULL, 0) == 0) {
        ram_mem_size_bytes = size;
    }
#elif defined(_SC_AIX_REALMEM)
    ram_mem_size_bytes = sysconf(_SC_AIX_REALMEM) * (size_t)1024L;

#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
    ram_mem_size_bytes =  static_cast<size_t>(sysconf(_SC_PHYS_PAGES)) *
        static_cast<size_t>(sysconf(_SC_PAGE_SIZE));
#endif
    return ram_mem_size_bytes;
}

}  // namespace util
}  // namespace ov