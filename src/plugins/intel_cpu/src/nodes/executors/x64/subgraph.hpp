// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <set>
#include <vector>

#include "cache/multi_cache.h"
#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "emitters/snippets/x64/kernel_executors/brgemm_copy_b.hpp"
#include "nodes/executors/repacking_subgraph.hpp"

namespace ov::intel_cpu {

class SubgraphExecutor : public SubgraphRepackingExecutor<BrgemmCopyBKernel> {
public:
    using SubgraphRepackingExecutor<BrgemmCopyBKernel>::SubgraphRepackingExecutor;

#ifdef SNIPPETS_DEBUG_CAPS
    SubgraphExecutor(const std::shared_ptr<CPURuntimeConfig>& snippet_config,
                     const std::shared_ptr<SubgraphAttrs>& snippet_attrs,
                     const std::shared_ptr<SubgraphCodeGenerator>& snippet,
                     const std::vector<ptrdiff_t>& start_offset_in,
                     const std::vector<ptrdiff_t>& start_offset_out,
                     const BufferScratchpadAllocator& allocator,
                     const ov::intel_cpu::MultiCacheWeakPtr& kernel_cache);

protected:
    void segfault_detector() const override;

private:
    bool enabled_segfault_detector = false;
#endif
};

class SubgraphStaticExecutor : public SubgraphRepackingStaticExecutor<SubgraphExecutor> {
public:
    using SubgraphRepackingStaticExecutor<SubgraphExecutor>::SubgraphRepackingStaticExecutor;
};

class SubgraphDynamicSpecializedExecutor : public SubgraphRepackingDynamicSpecializedExecutor<SubgraphExecutor> {
public:
    using SubgraphRepackingDynamicSpecializedExecutor<SubgraphExecutor>::SubgraphRepackingDynamicSpecializedExecutor;
};

}  // namespace ov::intel_cpu
