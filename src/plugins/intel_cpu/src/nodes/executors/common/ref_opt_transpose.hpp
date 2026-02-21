// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/transpose.hpp"
#include "onednn/iml_type_mapper.h"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {
class RefOptimizedTransposeExecutor : public TransposeExecutor {
public:
    using TransposeExecutor::TransposeExecutor;

    static bool supports(const TransposeConfig& config);
    static ExecutorPtr create(const TransposeAttrs& attrs,
                              [[maybe_unused]] const MemoryArgs& memory,
                              const ExecutorContext::CPtr& context);
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) override;
    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::ref;
    }

private:
    bool init(const MemoryArgs& memory) override;
};

}  // namespace ov::intel_cpu
