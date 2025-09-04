// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kleidiai_mm.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "kai/kai_common.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp_qsi8cx_neon.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/executors/acl/acl_fullyconnected_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/debug_capabilities.h"
#include "utils/precision_support.h"

#define FLOAT_MAX 3.4028235e38f
#define FLOAT_MIN (-3.4028235e38f)

namespace ov::intel_cpu {

using namespace executor;
using namespace ov::element;

template <typename T>
static std::vector<T> normalizeDimsTo2D(const std::vector<T>& dims) {
    return {std::accumulate(dims.begin(), dims.end() - 1, static_cast<T>(1), std::multiplies<T>()),
            dims[dims.size() - 1]};
}

static bool useDynamicQuantizationImpl(const FCAttrs& attrs, const MemoryDescPtr& weightDesc) {
    if (attrs.dynamicQuantizationGroupSize != std::numeric_limits<uint64_t>::max()) {
        return false;
    }

    if (!hasIntDotProductSupport() || !hasInt8MMSupport()) {
        return false;
    }

    return weightDesc->getPrecision() == element::i8;
}

bool MatMulKleidiAIExecutor::supports(const FCConfig& config) {
    return config.descs.at(ARG_WEI)->getPrecision() == element::f32 ||
           useDynamicQuantizationImpl(config.attrs, config.descs.at(ARG_WEI));
}

MatMulKleidiAIExecutor::MatMulKleidiAIExecutor(const FCAttrs& attrs,
                                               const MemoryArgs& memory,
                                               const ExecutorContext::CPtr& context)
    : m_attrs(attrs),
      m_memoryArgs(memory) {
    auto srcMem = memory.at(ARG_SRC);
    auto weiMem = memory.at(ARG_WEI);
    auto weiDims = weiMem->getDesc().getShape().getDims();
    auto N = weiDims[0];
    auto K = weiDims[1];

    if (memory.at(ARG_BIAS)->getDataAs<float>() == nullptr) {
        auto biasDesc = std::make_shared<CpuBlockedMemoryDesc>(f32, Shape({N}));
        biasMem = std::make_shared<Memory>(context->getEngine(), biasDesc);
        biasMem->nullify();
    } else {
        biasMem = memory.at(ARG_BIAS);
    }
    if (memory.at(ARG_SRC)->getPrecision() != memory.at(ARG_WEI)->getPrecision()) {
        aclfcAttrs.isConvertedWeights = true;
    }
    auto originalWeightsDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& wgtDims = originalWeightsDesc->getShape().getStaticDims();
    const VectorDims wgtDims2D = reshapeDownToRank<2>(wgtDims);
    originalWeightsDesc = std::make_shared<CpuBlockedMemoryDesc>(originalWeightsDesc->getPrecision(), Shape{wgtDims2D});
    auto dnnlSrcDesc = MemoryDescUtils::convertToDnnlMemoryDesc(originalWeightsDesc);

    // Whether dynamic quantization is enabled
    useDynamicQuant = useDynamicQuantizationImpl(attrs, originalWeightsDesc);

    if (!useDynamicQuant) {
        auto dstDesc = originalWeightsDesc->cloneWithNewPrecision(memory.at(ARG_SRC)->getDescPtr()->getPrecision());
        auto dnnlDstDesc = MemoryDescUtils::convertToDnnlMemoryDesc(dstDesc);
        packedWeights = acl_fc_executor::reorderWeights(memory, context, aclfcAttrs, dnnlSrcDesc, dnnlDstDesc);

        const size_t rhsPackedSize = kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(N, K);
        auto rhsPackedDesc = std::make_shared<CpuBlockedMemoryDesc>(u8, Shape({rhsPackedSize}));
        rhsPackedMem = std::make_shared<Memory>(context->getEngine(), rhsPackedDesc);

        auto* bias = biasMem->getDataAs<float>();
        auto* rhs_packed = static_cast<float*>(rhsPackedMem->getData());
        auto* rhs = static_cast<float*>(packedWeights->getData());
        const size_t rhs_stride = N * sizeof(float);

        const size_t nr = ukernel_f32.get_nr();
        const size_t kr = ukernel_f32.get_kr();
        const size_t sr = ukernel_f32.get_sr();

        kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(1,
                                                         N,
                                                         K,
                                                         nr,
                                                         kr,
                                                         sr,          // Packing arguments
                                                         rhs_stride,  // RHS stride
                                                         rhs,         // RHS
                                                         bias,        // Bias
                                                         nullptr,     // Scale
                                                         rhs_packed,  // RHS packed
                                                         0,
                                                         nullptr);
    } else {
        ukernel_i8 = hasInt8MMSupport() ? &ukernel_i8_imm : &ukernel_i8_dotprod;
        MemoryPtr weightsMemory = memory.at(ARG_WEI);
        if (!attrs.weightsNonTransposed) {
            auto dnnlSrcDesc = MemoryDescUtils::convertToDnnlMemoryDesc(originalWeightsDesc);
            auto dnnlDstDesc = acl_fc_executor::makeTransposedWeightDescriptor(dnnlSrcDesc, dnnlSrcDesc);
            weightsMemory = acl_fc_executor::reorderData(dnnlSrcDesc, dnnlDstDesc, memory.at(ARG_WEI), context);
        }

        mr = ukernel_i8->get_mr();
        nr = ukernel_i8->get_nr();
        kr = ukernel_i8->get_kr();
        sr = ukernel_i8->get_sr();

        auto* bias = biasMem->getDataAs<float>();
        auto* rhs_native_qs8cx = weightsMemory->getDataAs<int8_t>();
        float* rhs_scales = static_cast<float*>(memory.at(ARG_WEI | ARG_ATTR_SCALES)->getData());

        const size_t rhsPackedSize = kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(N, K, nr, kr, sr);
        auto rhsPackedDesc = std::make_shared<CpuBlockedMemoryDesc>(i8, Shape({rhsPackedSize}));
        rhsPackedMem = std::make_shared<Memory>(context->getEngine(), rhsPackedDesc);
        auto* rhs_packed_qs8cx = static_cast<int8_t*>(rhsPackedMem->getData());

        kai_rhs_pack_qsi8cx_params params{};
        params.lhs_zero_point = 1;

        kai_run_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(1,
                                                 N,
                                                 K,
                                                 nr,
                                                 kr,
                                                 sr,
                                                 rhs_native_qs8cx,
                                                 bias,
                                                 rhs_scales,
                                                 rhs_packed_qs8cx,
                                                 0,
                                                 &params);

        // Create scratchpad to initialize memory for LHS in update()
        scratchPad = context->getScratchPad();
    }
}

bool MatMulKleidiAIExecutor::update(const MemoryArgs& memory) {
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();
    const auto& wgtDims = weiDesc->getShape().getStaticDims();
    // Weights are transposed by MatMulConstTransposesExtraction
    // K is the IC of weight
    // the weight is reshaped to [-1, K] in ConvertMatMulToFC
    K = wgtDims[1];
    N = wgtDims[0];

    const auto& outDims = dstDesc->getShape().getStaticDims();
    if (outDims.size() > 2) {
        M = std::accumulate(outDims.begin(), outDims.end() - 1, 1, std::multiplies<>());
    } else {
        M = outDims[0];
    }
    // Assign LHS memory
    if (useDynamicQuant) {
        const size_t _m_blocks = (M + BLOCK_SIZE_M_INT8 - 1) / BLOCK_SIZE_M_INT8;
        packed_lhs_block_in_bytes_int8 =
            kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(BLOCK_SIZE_M_INT8, K, mr, kr, sr);
        const size_t lhsPackedSize = packed_lhs_block_in_bytes_int8 * _m_blocks;
        auto lhsPackedDesc = std::make_shared<CpuBlockedMemoryDesc>(i8, Shape({lhsPackedSize}));
        lhsPackedMem = scratchPad->createScratchPadMem(lhsPackedDesc);
    }
    return true;
}

void MatMulKleidiAIExecutor::execute(const MemoryArgs& memory) {
    auto srcMem = memory.at(ARG_SRC);
    auto weiMem = memory.at(ARG_WEI);
    auto dstMem = memory.at(ARG_DST);
    auto srcDims = normalizeDimsTo2D(srcMem->getDesc().getShape().getDims());
    auto weiDims = weiMem->getDesc().getShape().getDims();
    auto M = srcDims[0];
    auto K = srcDims[1];
    auto N = weiDims[0];

    const size_t lhs_stride = K * sizeof(float);
    const size_t dst_stride_row = N * sizeof(float);
    const size_t dst_stride_col = sizeof(float);
    auto* lhs = srcMem->getDataAs<float>();
    auto* dst = dstMem->getDataAs<float>();

    size_t n_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (!useDynamicQuant) {
        auto* rhs_packed = static_cast<float*>(rhsPackedMem->getData());

        parallel_for(n_blocks, [&](size_t n_block) {
            size_t n_start = (n_block * BLOCK_SIZE);
            size_t n_end = std::min(n_start + BLOCK_SIZE, N);
            size_t n_block_size = n_end - n_start;
            const size_t rhs_packed_offset = ukernel_f32.get_rhs_packed_offset(n_start, K);
            const size_t dst_offset = ukernel_f32.get_dst_offset(0, n_start, dst_stride_row);
            const float* rhs_ptr = (rhs_packed + rhs_packed_offset / sizeof(float));
            float* dst_ptr = (dst + dst_offset / (sizeof(float)));
            ukernel_f32.run_matmul(M,
                                   n_block_size,
                                   K,
                                   lhs,
                                   lhs_stride,
                                   rhs_ptr,
                                   dst_ptr,
                                   dst_stride_row,
                                   dst_stride_col,
                                   FLOAT_MIN,
                                   FLOAT_MAX);
        });
    } else {
        // Create packed LHS and RHS
        auto* lhs_packed_qa8dx = lhsPackedMem->getDataAs<int8_t>();
        auto* rhs_packed_qs8cx = rhsPackedMem->getDataAs<int8_t>();

        constexpr size_t m_step = BLOCK_SIZE_M_INT8;
        constexpr size_t n_step = 4;
        const size_t M_BLOCKS = (M + m_step - 1) / m_step;
        const size_t N_BLOCKS = (N + n_step - 1) / n_step;
        const size_t lhs_packed_offset = ukernel_i8->get_lhs_packed_offset(0, K);

        parallel_for(M_BLOCKS, [&](size_t m_blk) {
            const size_t M_iter = std::min(M - m_blk * m_step, m_step);
            auto* lhs_packed_qa8dx_B = lhs_packed_qa8dx + m_blk * packed_lhs_block_in_bytes_int8;

            kai_run_lhs_quant_pack_qai8dxp_f32(M_iter,
                                               K,
                                               mr,
                                               kr,
                                               sr,
                                               0,
                                               lhs + m_blk * m_step * K,  // LHS (F32)
                                               lhs_stride,
                                               lhs_packed_qa8dx_B  // lhs packed output
            );
            parallel_for(N_BLOCKS, [&](size_t n_blk) {
                //  matmul exec
                const size_t rhs_packed_offset = ukernel_i8->get_rhs_packed_offset(n_blk * n_step, K);
                const size_t dst_offset = ukernel_i8->get_dst_offset(m_blk * m_step, n_blk * n_step, dst_stride_row);
                const void* rhs_ptr = static_cast<const void*>(rhs_packed_qs8cx + rhs_packed_offset);
                const auto* lhs_ptr = static_cast<const void*>(lhs_packed_qa8dx_B + lhs_packed_offset);
                float* dst_ptr = (dst + dst_offset / sizeof(float));
                const size_t N_iter = std::min(N - n_blk * n_step, n_step);
                ukernel_i8->run_matmul(M_iter,
                                       N_iter,
                                       K,
                                       lhs_ptr,
                                       rhs_ptr,
                                       dst_ptr,
                                       dst_stride_row,
                                       dst_stride_col,
                                       FLOAT_MIN,
                                       FLOAT_MAX);
            });
        });
    }
}

void MatMulKleidiAIExecutor::moveMemToNumaNode(int numaNodeID) {
    if (curNumaNode == numaNodeID) {
        return;
    }
    curNumaNode = numaNodeID;
    mbind_move(packedWeights, numaNodeID);
    if (m_attrs.withBias) {
        mbind_move(m_memoryArgs.at(ARG_BIAS), numaNodeID);
    }
}

}  // namespace ov::intel_cpu
