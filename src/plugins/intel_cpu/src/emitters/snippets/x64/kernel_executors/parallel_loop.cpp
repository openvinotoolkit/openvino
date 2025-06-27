// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "parallel_loop.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "common/utils.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/parallel.hpp"

namespace ov::intel_cpu {
static inline uintptr_t* apply_byte_offset(uintptr_t* ptr, size_t offset) {
    // Note: we need to cast to char* to allow for arbitrary pointer shifts
    return reinterpret_cast<uintptr_t*>(reinterpret_cast<char*>(ptr) + offset);
}

size_t ParallelLoopConfig::hash() const {
    return dnnl::impl::hash_combine(0, m_increment);
}

void ParallelLoopExecutor::update_kernel([[maybe_unused]] const ParallelLoopConfig& c,
                                         std::shared_ptr<ParallelLoopKernel>& kernel) const {
    if (kernel == nullptr) {
        kernel = std::make_shared<ParallelLoopKernel>();
    }
}

void ParallelLoopExecutor::execute(const ParallelLoopExecutor* executor, call_args* call_args) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    const auto& config = static_cast<const ParallelLoopConfig&>(executor->get_config());
    const auto& loop_args = call_args->loop_args;

    const auto increment = config.get_increment();
    const int num_chunks = loop_args->m_work_amount / increment;
    const int nthr = std::min(parallel_get_max_threads(), num_chunks);

    // todo: it might worth to use num_ptrs as a template parameter, because it is always known in advance
    //  plus it would enable additional compiler optimizations like vectorized mem copy and for loops
    const auto num_ptrs = loop_args->m_num_data_ptrs;
    const auto& ptr_increments = loop_args->m_ptr_increments;

    const auto& stack_ptr = call_args->mem_ptrs;
    parallel_nt_static(nthr, [&](const int ithr, const int nthr) {
        int start_chunk = 0, end_chunk = 0;
        splitter(num_chunks, nthr, ithr, start_chunk, end_chunk);

        std::vector<uintptr_t*> mem_ptrs(num_ptrs);
        for (int i = 0; i < num_ptrs; i++) {
            mem_ptrs[i] = apply_byte_offset(stack_ptr[i], ptr_increments[i] * start_chunk);
        }

        const auto internal_seq_loop_work_amount = (end_chunk - start_chunk) * increment;
        call_args->preamble_ptr(internal_seq_loop_work_amount, reinterpret_cast<void*>(mem_ptrs.data()));
    });

    for (int64_t i = 0; i < num_ptrs; i++) {
        // Note: since we don't apply ptr_increments in the parallel section,
        // they are applied here together with finalization offsets
        const auto final_offset = ptr_increments[i] * num_chunks + loop_args->m_finalization_offsets[i];
        stack_ptr[i] = apply_byte_offset(stack_ptr[i], final_offset);
    }
}

}  // namespace ov::intel_cpu
