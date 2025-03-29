// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
    template <typename T, typename... Args>
    SubgraphStaticExecutor(T&& first, Args&&... rest)
        : SubgraphExecutor(std::forward<T>(first), std::forward<Args>(rest)...),
          SubgraphStaticBaseExecutor() {}

    void exec_impl(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;
};

class SubgraphDynamicSpecializedExecutor : public SubgraphExecutor, public SubgraphDynamicSpecializedBaseExecutor {
public:
    template <typename T, typename... Args>
    SubgraphDynamicSpecializedExecutor(T&& first, Args&&... rest)
        : SubgraphExecutor(std::forward<T>(first), std::forward<Args>(rest)...),
          SubgraphDynamicSpecializedBaseExecutor(std::forward<T>(first)) {}

    void exec_impl(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override;
};

}  // namespace ov::intel_cpu
