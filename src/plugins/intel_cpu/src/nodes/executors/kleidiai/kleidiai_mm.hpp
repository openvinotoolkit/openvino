// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>

#include "arm_neon.h"
#include "cpu_memory.h"
#include "nodes/executors/acl/acl_fullyconnected_utils.hpp"
#include "nodes/executors/fullyconnected_config.hpp"

#include "kleidiai_common.hpp"

namespace ov::intel_cpu {

class MatMulKleidiAIExecutor : public Executor {
public:
    MatMulKleidiAIExecutor(const FCAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);

    void execute(const MemoryArgs& memory) override;

    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::kleidiai;
    }

    // offloads execution data preparation from the exec call
    bool update(const MemoryArgs& memory) override;

    static bool supports(const FCConfig& config);

    void moveMemToNumaNode(int numaNodeID) override;

    bool isGroupQuantizationEnabled(const FCAttrs& attrs, const MemoryArgs& memory);

private:
    DnnlScratchPadPtr scratchPad;
    ACLFCAttrs aclfcAttrs;
    MemoryPtr biasMem;
    MemoryPtr rhsPackedMem;
    MemoryPtr lhsPackedMem;
    size_t M = 0UL, N = 0UL, K = 0UL;
    ExecutorContext::CPtr executorContext;
    std::shared_ptr<kai_common::uKernelBase> _kernel;

};

using MatMulKleidiAIExecutorPtr = std::shared_ptr<MatMulKleidiAIExecutor>;

}  // namespace ov::intel_cpu
