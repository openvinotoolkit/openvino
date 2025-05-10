// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "thread_pool_imp.hpp"

namespace ov::intel_cpu {

dnnl::stream make_stream(const dnnl::engine& engine, std::shared_ptr<ThreadPool> thread_pool) {
    dnnl::stream stream;
#if OV_THREAD == OV_THREAD_TBB
    static auto g_cpu_parallel = std::make_shared<CpuParallel>(ov::hint::TbbPartitioner::STATIC, 32);
    static auto g_thread_pool = std::make_shared<ThreadPool>(g_cpu_parallel);
    auto& cur_thread_pool = thread_pool;
    if (!thread_pool) {
        cur_thread_pool = g_thread_pool;
    }
    stream = dnnl::threadpool_interop::make_stream(engine, cur_thread_pool.get());
#    if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    dnnl::impl::threadpool_utils::activate_threadpool(cur_thread_pool.get());
#    endif
#else
    stream = dnnl::stream(engine);
#endif
    return stream;
}

}  // namespace ov::intel_cpu
