// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/gathermatmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

class GatherMatmulDnnlExecutor : public Executor {
public:
    static bool supports(const GatherMatmulConfig& config);

    GatherMatmulDnnlExecutor(const GatherMatmulAttrs& attrs,
                             const MemoryArgs& memory,
                             const ExecutorContext::CPtr& context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override;

private:
    class InnerProduct;
    using InnerProductPtr = std::shared_ptr<InnerProduct>;

    ExecutorContext::CPtr m_context;

    MemoryPtr m_weightsMemory;
    MemoryPtr m_scalesMemory;
    MemoryPtr m_zpMemory;

    InnerProductPtr m_gemvImpl;
    InnerProductPtr m_gemmImpl;

    MemoryPtr m_tmpInpBuffer;
    MemoryDescPtr m_tmpInputDesc;
    MemoryDescPtr m_tmpOutputDesc;

    bool m_bf16AmxMode = false;
    bool m_withBias = false;
    impl_desc_type m_implType = impl_desc_type::unknown;
};

}  // namespace ov::intel_cpu
