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
    auto hash = dnnl::impl::hash_combine(0, m_loop_args.m_work_amount);
    hash = dnnl::impl::hash_combine(hash, m_loop_args.m_num_data_ptrs);
    // Note: including ptrs in the hash => every instance will be unique
    hash = dnnl::impl::hash_combine(hash, m_loop_args.m_finalization_offsets);
    hash = dnnl::impl::hash_combine(hash, m_loop_args.m_ptr_increments);
    hash = dnnl::impl::hash_combine(hash, m_loop_args.m_num_data_ptrs);
    hash = dnnl::impl::hash_combine(hash, m_increment);
    return hash;
}

void ParallelLoopExecutor::update_kernel([[maybe_unused]] const ParallelLoopConfig& c,
                                         std::shared_ptr<ParallelLoopKernel>& kernel) const {
    kernel = std::make_shared<ParallelLoopKernel>();
}

void ParallelLoopExecutor::execute(const ParallelLoopExecutor* executor,
                                   uintptr_t** stack_ptr,
                                   loop_preamble_t preamble_ptr) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    const auto& config = static_cast<const ParallelLoopConfig&>(executor->get_config());
    const auto& loop_args = config.get_loop_args();

    const auto increment = config.get_increment();
    int num_chunks = loop_args.m_work_amount / increment;
    // todo: do we need to pass num_threads through config?
    // int num_threads = std::getenv("N") ? std::atoi(std::getenv("N")) : parallel_get_max_threads();
    int nthr = std::min(parallel_get_max_threads(), num_chunks);

    // todo: it might worth to use num_ptrs as a template parameter, because it is always known in advance
    //  plus it would enable additional compiler optimizations like vectorized mem copy and for loops
    const auto num_ptrs = loop_args.m_num_data_ptrs;
    const auto& ptr_increments = loop_args.m_ptr_increments;
    const auto& dtype_sizes = loop_args.m_dtype_sizes;
    parallel_nt_static(nthr, [&](const int ithr, const int nthr) {
        decltype(num_chunks) start_chunk = 0, end_chunk = 0;
        splitter(num_chunks, nthr, ithr, start_chunk, end_chunk);

        const auto start = start_chunk * increment;
        const auto end = end_chunk * increment;

        std::vector<uintptr_t*> mem_ptrs;
        mem_ptrs.reserve(num_ptrs);
        for (int i = 0; i < num_ptrs; i++) {
            const auto stack_ptr_offset = ptr_increments[i] * dtype_sizes[i] * start;
            mem_ptrs.push_back(apply_byte_offset(stack_ptr[i], stack_ptr_offset));
        }
        auto* updated_ptrs = reinterpret_cast<void*>(mem_ptrs.data());
        preamble_ptr(end - start, updated_ptrs);
    });
    // todo: we can precompute these ptr shifts when loop_info_t is created
    for (int i = 0; i < num_ptrs; i++) {
        // Note: since we don't apply ptr_increments in the parallel section,
        // they are applied here together with finalization offsets
        const auto final_offset =
            (ptr_increments[i] * loop_args.m_work_amount + loop_args.m_finalization_offsets[i]) * dtype_sizes[i];
        stack_ptr[i] = apply_byte_offset(stack_ptr[i], final_offset);
    }
}

}  // namespace ov::intel_cpu
