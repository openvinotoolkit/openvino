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
 * @param dest
 * pointer to the object to copy to
 * @param destsz
 * max number of bytes to modify in the destination (typically the size
 * of the destination object)
 * @param src
 pointer to the object to copy from
 * @param count
 number of bytes to copy
 @return zero on success and non-zero value on error.
 */

inline int simple_copy(void* dest, size_t destsz, void const* src, size_t count) {
    size_t i;
    if (!src || count > destsz ||
        count > (dest > src ? ((uintptr_t)dest - (uintptr_t)src)
                            : ((uintptr_t)src - (uintptr_t)dest))) {
        // zero out dest if error detected
        memset(dest, 0, destsz);
        return -1;
    }

    for (i = 0; i < count; ++i) (reinterpret_cast<uint8_t*>(dest))[i] = (reinterpret_cast<const uint8_t*>(src))[i];
    return 0;
}
