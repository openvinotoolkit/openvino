// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "cpu_memory.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/selective_ssm_config.hpp"

namespace ov::intel_cpu::kernel {
class JitKernelBase;
}

namespace ov::intel_cpu {

class SelectiveSSMJitExecutor : public Executor {
public:
    static bool supports(const SelectiveSSMConfig& config);

    SelectiveSSMJitExecutor(const SelectiveSSMAttrs& attrs, const MemoryArgs& memory, ExecutorContext::CPtr context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override;

private:
    bool update_scratchpad(const MemoryArgs& memory, size_t state_size, size_t block_head_dim);

    ExecutorContext::CPtr m_context;
    MemoryPtr m_state_scratch;
    std::shared_ptr<kernel::JitKernelBase> m_jit_kernel;
    std::shared_ptr<kernel::JitKernelBase> m_decode_jit_kernel;
    size_t m_cached_state_size = 0;
    size_t m_cached_sequence_length = 0;
    size_t m_block_head_dim = 0;
    size_t m_cached_scratch_head_dim = 0;
    uint8_t m_cached_data_type = 0xFF;
};

}  // namespace ov::intel_cpu
