// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "cpu_memory.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/gated_delta_net_config.hpp"

namespace ov::intel_cpu::kernel {
class JitKernelBase;
}

namespace ov::intel_cpu {

class GdnJitExecutor : public Executor {
public:
    static bool supports(const GatedDeltaNetConfig& config);

    GdnJitExecutor(const GatedDeltaNetAttrs& attrs, const MemoryArgs& memory, ExecutorContext::CPtr context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override;

private:
    bool updateKernelAndScratchpad(const MemoryArgs& memory);

    GatedDeltaNetAttrs m_attrs;
    ExecutorContext::CPtr m_context;
    std::shared_ptr<kernel::JitKernelBase> m_jitKernel;
    MemoryPtr m_tmpInpBuffer;

    ov::element::Type m_cachedPrecision = ov::element::dynamic;
    size_t m_cachedHeadSize = 0;
    size_t m_cachedVTile = 0;
};

}  // namespace ov::intel_cpu
