// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/interpolate_config.hpp"
#include "nodes/executors/memory_arguments.hpp"

namespace ov::intel_cpu {

class InterpolateJitExecutorWrapper : public Executor {
public:
    InterpolateJitExecutorWrapper(InterpolateAttrs attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);
    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override;

private:
    InterpolateAttrs m_attrs;
    ExecutorContext::CPtr m_context;
    std::shared_ptr<InterpolateExecutorBase> m_jit;
    std::vector<int> m_conversion5DMap;
    std::vector<MemoryCPtr> m_fqMemory;
    std::vector<const void*> m_fqDataPtrs;
};

}  // namespace ov::intel_cpu
