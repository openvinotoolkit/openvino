// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "parallel_loop.hpp"

#include "openvino/core/parallel.hpp"
#include "common/utils.hpp"

namespace ov::intel_cpu {

size_t ParallelLoopConfig::hash() const {
    const auto hash = dnnl::impl::hash_combine(0, m_work_amount);
    return dnnl::impl::hash_combine(0, m_num_threads);
}

void ParallelLoopExecutor::execute(const ParallelLoopExecutor* executor, void* stack_ptr, loop_preamble_t preamble_ptr) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    const auto& config = static_cast<const ParallelLoopConfig&>(executor->get_config());

    int64_t work_amount = config.get_work_amount();
    int nthr = config.get_num_threads();

    parallel_nt_static(nthr, [&](const int ithr, const int nthr) {
        int64_t start = 0, end = 0;
        splitter(work_amount, nthr, ithr, start, end);
        preamble_ptr(start, end, stack_ptr);
    });
}


}  // namespace ov::intel_cpu
