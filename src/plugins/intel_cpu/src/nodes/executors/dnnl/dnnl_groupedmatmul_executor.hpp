// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_inner_product_gemm.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/groupedmatmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

// Executes GroupedMatMul-17 (and its compressed flavour) as a loop of oneDNN inner_product calls,
// one per group. CPU oneDNN has no grouped gemm primitive, but unlike GatherMatmul the rows of each
// group are guaranteed contiguous, so the rows are addressed in place instead of being gathered.
class GroupedMatMulDnnlExecutor : public Executor {
public:
    static bool supports(const GroupedMatMulConfig& config);

    GroupedMatMulDnnlExecutor(const GroupedMatMulAttrs& attrs,
                              const MemoryArgs& memory,
                              const ExecutorContext::CPtr& context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override;

private:
    using InnerProductPtr = dnnl_utils::InnerProductPtr;

    ExecutorContext::CPtr m_context;

    MemoryPtr m_weightsMemory;
    MemoryPtr m_scalesMemory;
    MemoryPtr m_zpMemory;

    InnerProductPtr m_gemvImpl;
    InnerProductPtr m_gemmImpl;

    MemoryPtr m_tmpInpBuffer;
    MemoryDescPtr m_tmpInputDesc;
    MemoryDescPtr m_tmpOutputDesc;

    // true for the 3D x 3D form: mat_a is [G, M, K] and every group owns exactly M rows.
    // false for the 2D x 3D form: mat_a is [T, K] and the row ranges come from the offsets input.
    bool m_isBatched = false;
    bool m_bf16AmxMode = false;
    impl_desc_type m_implType = impl_desc_type::unknown;
};

}  // namespace ov::intel_cpu
