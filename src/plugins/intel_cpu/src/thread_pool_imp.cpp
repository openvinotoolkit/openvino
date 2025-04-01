// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "thread_pool_imp.hpp"

namespace ov::intel_cpu {

threadpool_t* get_thread_pool() {
    static threadpool_t thread_pool(parallel_get_max_threads());
    return &thread_pool;
}

}  // namespace ov::intel_cpu