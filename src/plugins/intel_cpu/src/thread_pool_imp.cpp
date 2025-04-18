// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "thread_pool_imp.hpp"

namespace ov::intel_cpu {

dnnl::threadpool_interop::threadpool_iface* get_thread_pool() {
    if (get_tbb_partitioner() == TBB_STATIC) {
        static threadpool_tbb_static thread_pool(parallel_get_max_threads());
        return &thread_pool;
    } else {
        static threadpool_tbb_auto thread_pool(parallel_get_max_threads());
        return &thread_pool;
    }
}

int dnnl_get_multiplier() {
    return 32;
}

}  // namespace ov::intel_cpu
