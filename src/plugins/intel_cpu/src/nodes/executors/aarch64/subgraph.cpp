// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/aarch64/subgraph.hpp"

#if defined(SNIPPETS_DEBUG_CAPS) && (defined(__linux__) || defined(__APPLE__))

#    include <csignal>
#    include <cstddef>
#    include <iostream>
#    include <memory>
#    include <mutex>
#    include <string>
#    include <vector>

#    include "cache/multi_cache.h"
#    include "emitters/snippets/aarch64/cpu_generator.hpp"
#    include "emitters/snippets/aarch64/jit_segfault_detector_emitter.hpp"
#    include "emitters/snippets/aarch64/kernel_executors/gemm_copy_b.hpp"
#    include "emitters/snippets/cpu_runtime_configurator.hpp"
#    include "nodes/executors/repacking_subgraph.hpp"
#    include "nodes/executors/subgraph.hpp"
#    include "openvino/core/except.hpp"
#    include "openvino/core/parallel.hpp"

namespace ov::intel_cpu {

SubgraphExecutor::SubgraphExecutor(const std::shared_ptr<CPURuntimeConfig>& snippet_config,
                                   const std::shared_ptr<SubgraphAttrs>& snippet_attrs,
                                   const std::shared_ptr<SubgraphCodeGenerator>& snippet,
                                   const std::vector<ptrdiff_t>& start_offset_in,
                                   const std::vector<ptrdiff_t>& start_offset_out,
                                   const BufferScratchpadAllocator& allocator,
                                   const ov::intel_cpu::MultiCacheWeakPtr& kernel_cache)
    : SubgraphRepackingExecutor<aarch64::GemmCopyBKernel>(snippet_config,
                                                          snippet_attrs,
                                                          snippet,
                                                          start_offset_in,
                                                          start_offset_out,
                                                          allocator,
                                                          kernel_cache) {
    const auto target = std::dynamic_pointer_cast<const aarch64::CPUTargetMachine>(
        snippet_attrs->snippet->get_generator()->get_target_machine());
    enabled_segfault_detector = target && target->debug_config.enable_segfault_detector;
}

// NOLINTBEGIN(misc-include-cleaner) bug in clang-tidy
void SubgraphExecutor::segfault_detector() const {
    static std::mutex err_print_lock;
    if (enabled_segfault_detector) {
        auto signal_handler =
            []([[maybe_unused]] int signal) {
                std::lock_guard<std::mutex> guard(err_print_lock);
                if (auto* segfault_detector_emitter =
                    ov::intel_cpu::g_custom_segfault_handler<
                        ov::intel_cpu::aarch64::jit_uni_segfault_detector_emitter>
                        ->local()) {
                    std::cout << segfault_detector_emitter->info() << '\n';
                }
                auto tid = parallel_get_thread_num();
                OPENVINO_THROW("Segfault was caught by the signal handler in subgraph node execution on thread " +
                               std::to_string(tid));
            };
        struct sigaction new_handler {};
        new_handler.sa_handler = signal_handler;
        sigaction(SIGSEGV, &new_handler, nullptr);
    }
}
// NOLINTEND(misc-include-cleaner) bug in clang-tidy

}  // namespace ov::intel_cpu

#endif
