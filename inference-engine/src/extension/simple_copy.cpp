// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdint.h>
#include <string.h>
#include "simple_copy.h"

int simple_copy(void* dest, size_t destsz, void const* src, size_t count) {
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
