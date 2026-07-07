// Copyright (C) 2026 FUJITSU LIMITED
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/kleidiai/kleidiai_mm.hpp"
#include "nodes/executors/memory_arguments.hpp"

namespace ov::intel_cpu {
class GatherMatMulKleidiAIExecutor : public Executor {
public:
    static bool supports(const GatherMatmulConfig& config);

    GatherMatMulKleidiAIExecutor(const GatherMatmulAttrs& attrs,
                                 const MemoryArgs& memory,
                                 const ExecutorContext::CPtr& context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::kleidiai;
    }

private:
    ExecutorContext::CPtr m_context;
    size_t gather_axis_size = 0;
    std::vector<MatMulKleidiAIExecutorPtr> executor;
    std::vector<MemoryArgs> memArgs;
};
}  // namespace ov::intel_cpu