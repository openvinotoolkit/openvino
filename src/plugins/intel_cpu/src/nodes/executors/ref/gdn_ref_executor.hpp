// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <utility>

#include "cpu_memory.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/gated_delta_net_config.hpp"

namespace ov::intel_cpu {

class GdnRefExecutor : public Executor {
public:
    static bool supports(const GatedDeltaNetConfig& config);

    GdnRefExecutor(const GatedDeltaNetAttrs& attrs, const MemoryArgs& memory, ExecutorContext::CPtr context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override;

private:
    bool updateScratchpad(const MemoryArgs& memory);

    GatedDeltaNetAttrs m_attrs;
    ExecutorContext::CPtr m_context;
    MemoryPtr m_tmpInpBuffer;
    size_t m_cachedHeadSize = 0;
};

}  // namespace ov::intel_cpu
