// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/gathermatmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/type/element_type.hpp"

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

    // The GEMM for an input of the given number of rows, from m_gemmKey through the runtime cache. With
    // single_thread, a one-core GEMM that runs on the calling thread with a caller-provided scratchpad.
    [[nodiscard]] InnerProductPtr createGemm(size_t rows, bool single_thread = false) const;

    // Creates the one-core GEMMs on the first call and sizes the per-slot work areas for K and N. Returns false
    // when the expert-parallel path cannot run (no one-core implementation for the packed weights layout, or a
    // oneDNN runtime other than threadpool).
    bool initExpertParallel(size_t K, size_t N, const ov::element::Type& precision);

    // bf16 AMX, M > 1: cuts the rows routed to each gather index into tasks and runs every task with a one-core
    // GEMM, all tasks in one parallel region.
    void executeExpertParallel(const MemoryArgs& memory,
                               const std::vector<std::pair<int32_t, int32_t>>& gather_idx_map,
                               const std::vector<int32_t>& rows_per_gather_index,
                               size_t M);

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

    // Selects the row threshold below which an expert runs per-row GEMV instead of a GEMM, and allows the
    // expert-parallel path.
    bool m_bf16AmxMode = false;

    // Expert-parallel path (bf16 AMX). One one-core GEMM per task height (16, 32, 48, 64 rows), and m_epSlots work
    // areas of m_epSlotBytes in m_tmpInpBuffer: input rows, then output rows, then the GEMM scratchpad.
    bool m_expertParallelOk = false;
    bool m_epGemmsTried = false;
    std::vector<InnerProductPtr> m_epGemms;
    size_t m_epSlots = 0;
    size_t m_epSlotBytes = 0;
    size_t m_epInBytes = 0;
    size_t m_epOutBytes = 0;
    impl_desc_type m_implType = impl_desc_type::unknown;
};

}  // namespace ov::intel_cpu
