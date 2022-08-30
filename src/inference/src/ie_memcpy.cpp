// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_memcpy.h"

#include <stdint.h>
#include <string.h>

int ie_memcpy(void* dest, size_t destsz, void const* src, size_t count) {
    if (!src || count > destsz ||
        count > (dest > src ? ((uintptr_t)dest - (uintptr_t)src) : ((uintptr_t)src - (uintptr_t)dest))) {
        // zero out dest if error detected
        memset(dest, 0, destsz);
        return -1;
    }

    memcpy(dest, src, count);
    return 0;
}
