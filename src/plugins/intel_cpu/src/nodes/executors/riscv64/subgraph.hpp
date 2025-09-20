// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "cache/multi_cache.h"
#include "cpu_memory.h"
#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "nodes/executors/subgraph.hpp"

namespace ov::intel_cpu {

class SubgraphExecutor : public SubgraphBaseExecutor {
public:
    SubgraphExecutor(const std::shared_ptr<CPURuntimeConfig>& snippet_config,
                     const std::shared_ptr<SubgraphAttrs>& snippet_attrs,
                     const std::shared_ptr<SubgraphCodeGenerator>& snippet,
                     const std::vector<ptrdiff_t>& start_offset_in,
                     const std::vector<ptrdiff_t>& start_offset_out,
                     const BufferScratchpadAllocator& allocator,
                     const ov::intel_cpu::MultiCacheWeakPtr& kernel_cache);
};

class SubgraphStaticExecutor : public SubgraphExecutor, public SubgraphStaticBaseExecutor {
public:
    template <typename... Args>
    SubgraphStaticExecutor(const std::shared_ptr<ov::intel_cpu::CPURuntimeConfig>& config,
                           const std::set<size_t>& external_ptrs_idces,
                           size_t in_num,
                           Args&&... rest)
        : SubgraphExecutor(config, std::forward<Args>(rest)...),
          SubgraphStaticBaseExecutor(external_ptrs_idces, in_num) {}

    void exec_impl(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;
};

// Dynamic specialized executor is not used on RISCV64 yet, but keep the class for symmetry
class SubgraphDynamicSpecializedExecutor : public SubgraphExecutor, public SubgraphDynamicSpecializedBaseExecutor {
public:
    template <typename... Args>
    SubgraphDynamicSpecializedExecutor(const std::shared_ptr<ov::intel_cpu::CPURuntimeConfig>& config,
                                       const std::set<size_t>& external_ptrs_idces,
                                       size_t in_num,
                                       Args&&... rest)
        : SubgraphExecutor(config, std::forward<Args>(rest)...),
          SubgraphDynamicSpecializedBaseExecutor(config, external_ptrs_idces, in_num) {
        OPENVINO_THROW("SubgraphDynamicSpecializedExecutor is not supported on RISC-V platform");
    }

    void exec_impl(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;
};

}  // namespace ov::intel_cpu
