// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>

#include "arm_neon.h"
#include "cpu_memory.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"
#include "nodes/executors/acl/acl_fullyconnected_utils.hpp"
#include "nodes/executors/fullyconnected_config.hpp"

namespace ov {
namespace intel_cpu {

class MatMulKleidiAIExecutor : public Executor {
public:
    MatMulKleidiAIExecutor(const FCAttrs& attrs,
                           const PostOps& postOps,
                           const MemoryArgs& memory,
                           const ExecutorContext::CPtr& context);

    void execute(const MemoryArgs& memory) override;

    impl_desc_type implType() const override {
        return impl_desc_type::kleidiai;
    }

    // offloads execution data preparation from the exec call
    bool update(const MemoryArgs& memory) override;

    static bool supports(const FCConfig& config);

    void moveMemToNumaNode(int numaNodeID) override;

private:
    static constexpr kai_matmul_clamp_f32_f32_f32p_ukernel ukernel{
        kai_get_m_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
        kai_get_n_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
        kai_get_nr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
        kai_get_kr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
        kai_get_sr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
        kai_get_lhs_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
        kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
        kai_get_dst_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
        kai_get_dst_size_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
        kai_run_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla};

    const FCAttrs& m_attrs;
    ACLFCAttrs aclfcAttrs;
    const MemoryArgs& m_memoryArgs;
    MemoryPtr biasMem;
    MemoryPtr rhsPackedMem;
    MemoryCPtr packedWeights;
    int64_t M, N, K;
    int curNumaNode = -1;
};

using MatMulKleidiAIExecutorPtr = std::shared_ptr<MatMulKleidiAIExecutor>;

}  // namespace intel_cpu
}  // namespace ov