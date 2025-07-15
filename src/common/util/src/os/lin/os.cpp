// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/os.hpp"

#include <malloc.h>

#include <stdexcept>

namespace ov::util {
void set_mmap_threshold(int threshold) {
#if defined(M_MMAP_THRESHOLD)
    if (mallopt(M_MMAP_THRESHOLD, threshold) != 1) {
        throw std::runtime_error("Set M_MMAP_THRESHOLD failed");
    }
#endif
}
}  // namespace ov::util
