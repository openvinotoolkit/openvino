// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/aarch64/subgraph.hpp"

#include "snippets/op/subgraph.hpp"

namespace ov::intel_cpu {

SubgraphExecutor::SubgraphExecutor(const std::shared_ptr<CPURuntimeConfig>& snippet_config,
                                   const std::shared_ptr<SubgraphAttrs>& snippet_attrs,
                                   const std::shared_ptr<SubgraphCodeGenerator>& snippet,
                                   const std::vector<ptrdiff_t>& start_offset_in,
                                   const std::vector<ptrdiff_t>& start_offset_out,
                                   const BufferScratchpadAllocator& allocator,
                                   const ov::intel_cpu::MultiCacheWeakPtr& kernel_cache)
    : SubgraphBaseExecutor(snippet_config,
                           snippet_attrs,
                           snippet,
                           start_offset_in,
                           start_offset_out,
                           allocator,
                           kernel_cache) {
    m_buffer_scratchpad = allocator(m_internal_buffer_size);
}

void SubgraphStaticExecutor::exec_impl(const std::vector<MemoryPtr>& inMemPtrs,
                                       const std::vector<MemoryPtr>& outMemPtrs) {
    const auto& callable = m_schedule->get_callable<kernel>();

    auto initializer = [&](jit_snippets_call_args& call_args, size_t ithr) {
        init_call_args(call_args, inMemPtrs, outMemPtrs, m_start_offset_in, m_start_offset_out, ithr);
        update_scratchpad_ptr(call_args.buffer_scratchpad_ptr, ithr);
    };
    auto caller = [&](jit_snippets_call_args& call_args, const std::vector<size_t>& indexes, size_t ithr) {
        callable(&call_args, indexes.data());
    };

    if (m_parallel_exec_domain.size() == rank6D) {
        parallel_for6d(initializer, caller);
    } else {
        parallel_forNd(initializer, caller);
    }
}

void SubgraphDynamicSpecializedExecutor::exec_impl(const std::vector<MemoryPtr>& inMemPtrs,
                                                   const std::vector<MemoryPtr>& outMemPtrs) {
    const auto& callable = m_schedule->get_callable<dynamic_kernel>();

    OPENVINO_ASSERT(m_data_offsets.size() == inMemPtrs.size() + outMemPtrs.size(), "Incorrect data offset count!");
    OPENVINO_ASSERT(m_data_offsets.front().size() == m_parallel_exec_domain.size(),
                    "Data offsets with invalid ranks detected");

    // Note: we need to reset KernelExecutorTable to the state that was recorded in the
    // SubgraphDynamicSpecializedExecutor constructor because the table might've been used for other shapes
    m_reset_exec_table_state();

    std::vector<const uint8_t*> src_ptrs;
    std::vector<uint8_t*> dst_ptrs;
    init_original_ptrs(inMemPtrs, outMemPtrs, src_ptrs, dst_ptrs, m_start_offset_in, m_start_offset_out);

    auto initializer = [&](jit_snippets_call_args& call_args, size_t ithr) {
        init_call_args(call_args, ithr);
        update_scratchpad_ptr(call_args.buffer_scratchpad_ptr, ithr);
    };
    auto caller = [&](jit_snippets_call_args& call_args, const std::vector<size_t>& indexes, size_t ithr) {
        update_ptrs(call_args, src_ptrs, dst_ptrs, indexes);
        callable(&call_args);
    };

    if (m_parallel_exec_domain.size() == rank6D) {
        parallel_for6d(initializer, caller);
    } else {
        parallel_forNd(initializer, caller);
    }
}

}  // namespace ov::intel_cpu
