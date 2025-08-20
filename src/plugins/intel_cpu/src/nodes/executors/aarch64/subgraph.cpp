// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/aarch64/subgraph.hpp"

#include <csignal>
#include <mutex>
#include <string>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "cache/multi_cache.h"
#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "nodes/executors/subgraph.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#if defined(__linux__) && defined(SNIPPETS_DEBUG_CAPS)
#    include "emitters/snippets/aarch64/cpu_generator.hpp"
#    include "emitters/snippets/aarch64/jit_segfault_detector_emitter.hpp"
#endif

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

#if defined(__linux__) && defined(SNIPPETS_DEBUG_CAPS)
    const auto target = std::dynamic_pointer_cast<const ov::intel_cpu::aarch64::CPUTargetMachine>(
        snippet_attrs->snippet->get_generator()->get_target_machine());
    enabled_segfault_detector = target && target->debug_config.enable_segfault_detector;
#endif
}

void SubgraphStaticExecutor::exec_impl(const std::vector<MemoryPtr>& inMemPtrs,
                                       const std::vector<MemoryPtr>& outMemPtrs) {
    segfault_detector();
    const auto& callable = m_schedule->get_callable<kernel>();

    auto initializer = [&](jit_snippets_call_args& call_args, size_t ithr) {
        init_call_args(call_args, inMemPtrs, outMemPtrs, m_start_offset_in, m_start_offset_out);
        update_scratchpad_ptr(call_args.buffer_scratchpad_ptr, ithr);
    };
    auto caller =
        [&](jit_snippets_call_args& call_args, const std::vector<size_t>& indexes, [[maybe_unused]] size_t ithr) {
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
    segfault_detector();
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
        init_call_args(call_args);
        update_scratchpad_ptr(call_args.buffer_scratchpad_ptr, ithr);
    };
    auto caller =
        [&](jit_snippets_call_args& call_args, const std::vector<size_t>& indexes, [[maybe_unused]] size_t ithr) {
            update_ptrs(call_args, src_ptrs, dst_ptrs, indexes);
            callable(&call_args);
        };

    if (m_parallel_exec_domain.size() == rank6D) {
        parallel_for6d(initializer, caller);
    } else {
        parallel_forNd(initializer, caller);
    }
}

#if defined(__linux__) && defined(SNIPPETS_DEBUG_CAPS)
void SubgraphExecutor::segfault_detector() const {
    if (!enabled_segfault_detector)
        return;
    static std::mutex err_print_lock;
    __sighandler_t signal_handler = []([[maybe_unused]] int signal) {
        std::lock_guard<std::mutex> guard(err_print_lock);
        // Prefer emitter-provided info if available
        if (auto* e = ov::intel_cpu::aarch64::g_custom_segfault_handler->local()) {
            try {
                std::cout << e->info() << '\n';
            } catch (...) {
            }
        } else {
            try {
                std::cout << ov::intel_cpu::aarch64::get_segfault_tracking_info() << '\n';
            } catch (...) {
            }
        }
        auto tid = parallel_get_thread_num();
        OPENVINO_THROW(
            std::string("Segfault was caught by the signal handler in subgraph node execution on thread ") +
            std::to_string(tid));
    };
    struct sigaction new_handler {};
    new_handler.sa_handler = signal_handler;
    sigaction(SIGSEGV, &new_handler, nullptr);
}
#endif

}  // namespace ov::intel_cpu
