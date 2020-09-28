// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdlib.h>
#include "ie_api.h"

/**
 * @brief Copies bytes between buffers with security enhancements
 * Copies count bytes from src to dest. If the source and destination
 * overlap, the behavior is undefined.
 * @param dst
 * pointer to the object to copy to
 * @param dst_size
 * max number of bytes to modify in the destination (typically the size
 * of the destination object)
 * @param src
 pointer to the object to copy from
 * @param count
 number of bytes to copy
 @return zero on success and non-zero value on error.
 */

inline void cpu_memcpy(void* dst, const void* src, size_t count) {
#ifdef _WIN32
    memcpy_s(dst, count, src, count);
#else
    std::memcpy(dst, src, count);
#endif
}

inline int cpu_memcpy_s(void* dst, size_t dst_size, const void* src, size_t count) {
    size_t i;
    if (!src ||
        count > dst_size ||
        count > (dst > src ? ((uintptr_t)dst - (uintptr_t)src) : ((uintptr_t)src - (uintptr_t)dst))) {
        // zero out dest if error detected
        std::memset(dst, 0, dst_size);
        return -1;
    }

#ifdef _WIN32
    memcpy_s(dst, dst_size, src, count);
#else
    std::memcpy(dst, src, count);
#endif
    return 0;
}
