// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "cpu_memory.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/selective_ssm_config.hpp"

namespace ov::intel_cpu {

class SelectiveSSMExecutor : public Executor {
public:
    static bool supports(const SelectiveSSMConfig& config);

    SelectiveSSMExecutor(const SelectiveSSMAttrs& attrs, const MemoryArgs& memory, ExecutorContext::CPtr context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override;

private:
    bool update_scratchpad(const MemoryArgs& memory);

    ExecutorContext::CPtr m_context;
    MemoryPtr m_scratch;
    size_t m_scratch_head_dim = 0;
    size_t m_scratch_state_size = 0;
    size_t m_state_scratch_elements = 0;
    size_t m_projection_scratch_elements = 0;
    size_t m_cached_projection_elements = 0;
};

}  // namespace ov::intel_cpu
