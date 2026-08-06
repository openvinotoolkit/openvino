// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>

#include "arm_neon.h"
#include "cpu_memory.h"
#include "kleidiai_common.hpp"
#include "nodes/executors/acl/acl_fullyconnected_utils.hpp"
#include "nodes/executors/fullyconnected_config.hpp"

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

    void setKaiExecutorImplAsGatherMatmul();
    void set_gather_idx(const std::vector<std::pair<int32_t, int32_t>>& idxMap);

    enum kernelLookup : std::uint16_t {
        WEIGHT_FP32 = 1,
        WEIGHT_INT8 = 1 << 2,
        WEIGHT_INT4 = 1 << 3,
        ISA_DOTPROD = 1 << 4,
        ISA_I8MM = 1 << 5,
        QUANT_CHANNEL = 1 << 6,
        QUANT_GROUP = 1 << 7,
        QUANT_SYMMETRIC = 1 << 8,
        QUANT_ASYMMETRIC = 1 << 9
    };

private:
    static bool isGroupQuantizationEnabled(const MemoryArgs& memory);
    //  IMPL_TYPE :: Default
    //      [M, K] * [N, K] -> [M, N]
    //  IMPL_TYPE :: GatherMatmul
    //      [B, M, K] -> gather -> [M', K] * [N', K] -> scatter -> [B, N, K]
    enum class IMPL_TYPE : uint8_t { Default, GatherMatmul };
    static bool isAsymmetricQuantizationEnabled(const MemoryArgs& memory);
    DnnlScratchPadPtr scratchPad;
    IMPL_TYPE KaiExecutorImpl = IMPL_TYPE::Default;
    std::vector<std::pair<int32_t, int32_t>> gather_idx;
    MemoryDescPtr m_tmpInputDesc = nullptr;
    MemoryDescPtr m_tmpOutputDesc = nullptr;
    size_t lhsPackedSize = 0;
    ACLFCAttrs aclfcAttrs;
    MemoryPtr biasMem;
    MemoryPtr rhsPackedMem;
    MemoryPtr lhsPackedMem;
    size_t M = 0UL, N = 0UL, K = 0UL;
    ExecutorContext::CPtr executorContext;
    std::shared_ptr<kai_common::uKernelBase> _kernel;
    size_t kernelLookupKey = 0;
};

using MatMulKleidiAIExecutorPtr = std::shared_ptr<MatMulKleidiAIExecutor>;

}  // namespace ov::intel_cpu
