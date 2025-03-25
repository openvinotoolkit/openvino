// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>

#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"

namespace ov {
namespace intel_cpu {

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
    if (!src || count > dst_size ||
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

inline void cpu_parallel_memcpy(void* dst, const void* src, size_t count) {
    const size_t l2_cache_size = dnnl::utils::get_cache_size(2, true);
    if (count >= l2_cache_size) {
        auto src_int8 = static_cast<const uint8_t*>(src);
        auto dst_int8 = static_cast<uint8_t*>(dst);
        parallel_nt(0, [&](const size_t ithr, const size_t nthr) {
            size_t start = 0, end = 0;
            splitter(count, nthr, ithr, start, end);
            cpu_memcpy(dst_int8 + start, src_int8 + start, end - start);
        });
    } else {
        cpu_memcpy(dst, src, count);
    }
}

}  // namespace intel_cpu
}  // namespace ov
