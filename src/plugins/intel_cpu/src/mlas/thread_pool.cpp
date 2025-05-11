// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "thread_pool.hpp"

#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"

// This function impl the forward declaration in MLAS
size_t getCacheSizeMlas(int level, bool perCore) {
    return dnnl::utils::get_cache_size(level, perCore);
}

namespace ov {
namespace cpu {

size_t OVMlasThreadPool::DegreeOfParallelism() {
    // threadpool nullptr means single threaded
    return threadNum;
}

void OVMlasThreadPool::TrySimpleParallelFor(const std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn) {
    ov::parallel_nt_static(threadNum, [&](const size_t ithr, const size_t nthr) {
        std::ptrdiff_t start = 0, end = 0;
        ov::splitter(total, nthr, ithr, start, end);
        for (std::ptrdiff_t i = start; i < end; i++) {
            fn(i);
        }
    });
}
};  // namespace cpu
};  // namespace ov