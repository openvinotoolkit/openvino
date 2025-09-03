// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "thread_pool_imp.hpp"

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#if OV_THREAD == OV_THREAD_TBB_ADAPTIVE
#    include <oneapi/dnnl/dnnl_config.h>

#    include <common/dnnl_thread.hpp>
#    include <oneapi/dnnl/dnnl_threadpool.hpp>

#    include "cpu_parallel.hpp"
#    include "openvino/core/parallel.hpp"
#    include "openvino/runtime/intel_cpu/properties.hpp"
#endif

namespace ov::intel_cpu {

dnnl::stream make_stream(const dnnl::engine& engine, const std::shared_ptr<ThreadPool>& thread_pool) {  // NOLINT
    dnnl::stream stream;
#if OV_THREAD == OV_THREAD_TBB_ADAPTIVE
    static auto g_cpu_parallel = std::make_shared<CpuParallel>(ov::intel_cpu::TbbPartitioner::STATIC, 32);
    static auto g_thread_pool = std::make_shared<ThreadPool>(g_cpu_parallel);
    stream = dnnl::threadpool_interop::make_stream(engine, thread_pool ? thread_pool.get() : g_thread_pool.get());
#    if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    dnnl::impl::threadpool_utils::deactivate_threadpool();
    dnnl::impl::threadpool_utils::activate_threadpool(thread_pool ? thread_pool.get() : g_thread_pool.get());
#    endif
#else
    stream = dnnl::stream(engine);
#endif
    return stream;
}

}  // namespace ov::intel_cpu
