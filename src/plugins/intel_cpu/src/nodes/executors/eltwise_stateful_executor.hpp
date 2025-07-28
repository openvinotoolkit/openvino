// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/eltwise_executor.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/jit/eltwise.h"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

class EltwiseStatefulExecutor : public Executor {
public:
    EltwiseStatefulExecutor(EltwiseAttrs attrs, const MemoryArgs& memory, ExecutorContext::CPtr context);

    std::vector<VectorDims> updateInputBlockDims(const MemoryArgs& memory);

    // we can skip searching in the cache if broadcast policy for last input dims is not changed
    // last input dim == 1 means broadcasted (also if output dim == 1)
    // last input dim != 1 means not broadcasted
    bool canReuseCurrentExecutor(const std::vector<VectorDims>& dims_in);

    void updateExecutionParams(const std::vector<VectorDims>& inDims, const VectorDims& currentOutBlkDims);

    bool update(const MemoryArgs& memory) override;

    void execute(const MemoryArgs& memory) override;

    [[nodiscard]] impl_desc_type implType() const override;

private:
    EltwiseAttrs m_attrs;
    ExecutorContext::CPtr m_context;
    std::vector<ptrdiff_t> m_srcOffsets;
    ptrdiff_t m_dstOffset;

    // Shape agnostic support
    struct {
        VectorDims outDims;
        std::vector<VectorDims> inOffsets;
        VectorDims outOffsets;
    } m_execParams;

    std::vector<ov::element::Type> m_inpPrc;
    ov::element::Type m_outPrc;

    std::vector<MemoryCPtr> m_fqMemory;
    std::vector<const void*> m_fqDataPtrs;
    EltwiseShapeAgnosticData m_shapeAgnosticData;
    std::vector<VectorDims> m_currentInBlkDims;
    std::vector<bool> m_broadcastPolicy;
    EltwiseImplType eltwiseImplType = EltwiseImplType::reference;

    EltwiseExecutorPtr m_executor;
};
}  // namespace ov::intel_cpu
