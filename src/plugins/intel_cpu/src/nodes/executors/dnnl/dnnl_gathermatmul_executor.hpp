// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/gathermatmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

struct InnerProductKey;

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

    // The GEMM for an input of the given number of rows, from m_gemmKey through the runtime cache.
    [[nodiscard]] InnerProductPtr createGemm(size_t rows) const;

    ExecutorContext::CPtr m_context;

    MemoryPtr m_weightsMemory;
    MemoryPtr m_scalesMemory;
    MemoryPtr m_zpMemory;

    InnerProductPtr m_gemvImpl;
    // Key of the GEMMs run for M > 1 (experts with enough rows), each sized to the rows routed to its gather index
    // instead of the full input length. createGemm() sets the row count.
    std::shared_ptr<const InnerProductKey> m_gemmKey;

    MemoryPtr m_tmpInpBuffer;
    MemoryDescPtr m_tmpInputDesc;
    MemoryDescPtr m_tmpOutputDesc;

    // Selects the row threshold below which an expert runs per-row GEMV instead of a GEMM.
    bool m_bf16AmxMode = false;
    impl_desc_type m_implType = impl_desc_type::unknown;
};

}  // namespace ov::intel_cpu
