// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/os.hpp"

#include <stdexcept>

#if defined(OPENVINO_GNU_LIBC) && !defined(__ANDROID__)
#    include <malloc.h>
#endif

namespace ov::util {
#if defined(OPENVINO_GNU_LIBC) && !defined(__ANDROID__)
void set_mmap_threshold(int threshold) {
    if (mallopt(M_MMAP_THRESHOLD, threshold) != 1) {
        throw std::runtime_error("Set M_MMAP_THRESHOLD failed");
    }
}
#endif
}  // namespace ov::util
