// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "thread_pool.hpp"

#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"

namespace ov {
namespace cpu {
size_t DegreeOfParallelism(ThreadPool* tp) {
    // threadpool nullptr means single threaded
    return tp ? tp->threadNum : 1;
}
void TrySimpleParallelFor(ThreadPool* tp, const std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn) {
    if (tp == nullptr) {
        for (std::ptrdiff_t i = 0; i < total; i++) {
            fn(i);
        }
    } else {
        ov::parallel_nt(tp->threadNum, [&](const size_t ithr, const size_t nthr) {
            std::ptrdiff_t start = 0, end = 0;
            ov::splitter(total, nthr, ithr, start, end);
            for (std::ptrdiff_t i = start; i < end; i++) {
                fn(i);
            }
        });
    }
}
size_t getCacheSize(int level, bool perCore) {
    return dnnl::utils::get_cache_size(level, perCore);
}
};  // namespace cpu
};  // namespace ov