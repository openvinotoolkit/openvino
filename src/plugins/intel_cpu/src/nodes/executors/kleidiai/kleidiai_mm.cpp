// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kleidiai_mm.hpp"

#include <algorithm>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "kai/kai_common.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp_qsi8cx_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/acl/acl_fullyconnected_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"
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

    if (!hasIntDotProductSupport() && !hasInt8MMSupport()) {
        return false;
    }

    return weightDesc->getPrecision() == element::i8 || weightDesc->getPrecision() == element::i4;
}

bool MatMulKleidiAIExecutor::supports(const FCConfig& config) {
    return config.descs.at(ARG_WEI)->getPrecision() == element::f32 ||
           useDynamicQuantizationImpl(config.attrs, config.descs.at(ARG_WEI));
}

namespace {
// TODO: OffsetHelper is common util function. Move it to some common location
class OffsetHelper {
public:
    static OffsetHelper createOffsetHelper(const MemoryPtr& mem) {
        static const VectorDims empty_dims;
        std::bitset<2> broadcast_mask;
        if (nullptr == mem || mem->getDesc().empty()) {
            return {nullptr, empty_dims, broadcast_mask, 0};
        }
        return createOffsetHelper(*mem);
    }

    static OffsetHelper createOffsetHelper(const IMemory& mem) {
        std::bitset<2> broadcast_mask;
        auto* base_ptr = static_cast<uint8_t*>(mem.getData());
        auto desc = mem.getDescWithType<BlockedMemoryDesc>();
        const auto& strides = desc->getStrides();
        const auto prc = desc->getPrecision();
        const auto& shape = desc->getShape().getStaticDims();
        for (size_t i = 0; i < shape.size() && i < 2; i++) {
            if (shape[i] == 1) {
                broadcast_mask.set(i);
            }
        }
        return {base_ptr, strides, broadcast_mask, prc.bitwidth()};
    }

    void* operator()(size_t i0) const {
        if (!m_base_ptr) {
            return nullptr;
        }
        if (m_broadcast_mask.test(0)) {
            i0 = 0;
        }
        const size_t offset_bits = i0 * m_strides[0] * m_num_bits;
        const size_t offset = div_up(offset_bits, 8);  // 8 bits in byte
        return m_base_ptr + offset;
    }

    void* operator()(size_t i0, size_t i1) const {
        if (!m_base_ptr) {
            return nullptr;
        }
        if (m_broadcast_mask.test(0)) {
            i0 = 0;
        }
        if (m_broadcast_mask.test(1)) {
            i1 = 0;
        }
        const size_t offset_bits = i0 * m_strides[0] * m_num_bits + i1 * m_strides[1] * m_num_bits;
        const size_t offset = div_up(offset_bits, 8);  // 8 bits in byte
        return m_base_ptr + offset;
    }

    [[nodiscard]] void* get_base() const {
        return m_base_ptr;
    }

private:
    OffsetHelper(uint8_t* base_ptr, const VectorDims& strides, std::bitset<2> broadcast_mask, size_t num_bits)
        : m_base_ptr(base_ptr),
          m_strides(strides),
          m_num_bits(num_bits),
          m_broadcast_mask(broadcast_mask) {}

    uint8_t* m_base_ptr = nullptr;
    const VectorDims& m_strides;
    size_t m_num_bits;
    std::bitset<2> m_broadcast_mask;
};
}  // namespace

MatMulKleidiAIExecutor::MatMulKleidiAIExecutor(const FCAttrs& attrs,
                                               const MemoryArgs& memory,
                                               const ExecutorContext::CPtr& context)
    : executorContext(context) {
    auto srcMem = memory.at(ARG_SRC);
    auto weiMem = memory.at(ARG_WEI);
    auto weiDims = weiMem->getDesc().getShape().getDims();
    auto N = weiDims[0];
    auto K = weiDims[1];

    const bool hasBias = !memory.at(ARG_BIAS)->getDesc().empty();

    if (hasBias) {
        biasMem = memory.at(ARG_BIAS);
    } else {
        auto biasDesc = std::make_shared<CpuBlockedMemoryDesc>(f32, Shape({N}));
        biasMem = std::make_shared<Memory>(context->getEngine(), biasDesc);
        biasMem->nullify();
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
        if (!attrs.weightsNonTransposed) {
            dnnlDstDesc = acl_fc_executor::makeTransposedWeightDescriptor(dnnlDstDesc, dnnlSrcDesc);
            aclfcAttrs.isWeightsRepacked = true;
        }
        MemoryCPtr packedWeights =
            acl_fc_executor::reorderWeights(memory, context, aclfcAttrs, dnnlSrcDesc, dnnlDstDesc);

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
        MemoryPtr weightsMemory = memory.at(ARG_WEI);
        INT4_IMPL = weightsMemory->getDescPtr()->getPrecision() == element::i4;
        if (INT4_IMPL) {
            ukernel_i4 = hasInt8MMSupport() ? &ukernel_i4_imm : &ukernel_i4_dotprod;
            BLOCK_SIZE_M_LOWP = ukernel_i4->get_m_step();

            mr = ukernel_i4->get_mr();
            nr = ukernel_i4->get_nr();
            kr = ukernel_i4->get_kr();
            sr = ukernel_i4->get_sr();

            auto* rhs_native_qs4cx = weightsMemory->getDataAs<uint8_t>();
            float* rhs_scales = static_cast<float*>(memory.at(ARG_WEI | ARG_ATTR_SCALES)->getData());

            const size_t rhsPackedSize = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(N, K, nr, kr, sr);
            auto rhsPackedDesc = std::make_shared<CpuBlockedMemoryDesc>(i8, Shape({rhsPackedSize}));
            rhsPackedMem = std::make_shared<Memory>(context->getEngine(), rhsPackedDesc);
            auto* rhs_packed_qs4cx = static_cast<int8_t*>(rhsPackedMem->getData());

            kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params params{};
            params.lhs_zero_point = 1;
            params.rhs_zero_point = 0;

            auto* bias_ptr = hasBias ? biasMem->getDataAs<float>() : nullptr;
            if (attrs.weightsNonTransposed) {
                kai_run_rhs_pack_kxn_qsi4cxp_qs4cxs1s0(1,
                                                       N,
                                                       K,
                                                       nr,
                                                       kr,
                                                       sr,
                                                       rhs_native_qs4cx,
                                                       bias_ptr,
                                                       rhs_scales,
                                                       rhs_packed_qs4cx,
                                                       0,
                                                       &params);
            } else {
                kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(1,
                                                       N,
                                                       K,
                                                       nr,
                                                       kr,
                                                       sr,
                                                       rhs_native_qs4cx,
                                                       bias_ptr,
                                                       rhs_scales,
                                                       rhs_packed_qs4cx,
                                                       0,
                                                       &params);
            }

        } else {
            ukernel_i8 = hasInt8MMSupport() ? &ukernel_i8_imm : &ukernel_i8_dotprod;
            BLOCK_SIZE_M_LOWP = 16;
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
        }
        // Create scratchpad to initialize memory for LHS in update()
        scratchPad = context->getScratchPad();
    }
}

void MatMulKleidiAIExecutor::setKaiExecutorImplAsGatherMatmul() {
    KaiExecutorImpl = IMPL_TYPE::GatherMatmul;
}

void MatMulKleidiAIExecutor::set_gather_idx(const std::vector<std::pair<int32_t, int32_t>>& idxMap) {
    OPENVINO_ASSERT(KaiExecutorImpl == IMPL_TYPE::GatherMatmul,
                    "gather_idx is supported only for GatherMatmul Implementation");
    gather_idx = idxMap;
}

bool MatMulKleidiAIExecutor::update(const MemoryArgs& memory) {
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();
    const auto& wgtDims = weiDesc->getShape().getStaticDims();
    // Weights are transposed by MatMulConstTransposesExtraction
    // K is the IC of weight
    // the weight is reshaped to [-1, K] in ConvertMatMulToFC
    OPENVINO_ASSERT(wgtDims.size() == 2, "Weights Shape must be 2D");
    K = wgtDims[1];
    N = wgtDims[0];

    const auto& outDims = dstDesc->getShape().getStaticDims();
    if (KaiExecutorImpl == IMPL_TYPE::GatherMatmul && outDims.size() == 3) {
        M = outDims[1];
    } else if (outDims.size() > 2) {
        M = std::accumulate(outDims.begin(), outDims.end() - 1, 1, std::multiplies<>());
    } else {
        M = outDims[0];
    }
    // Assign LHS memory
    if (useDynamicQuant) {
        const size_t _m_blocks = (M + BLOCK_SIZE_M_LOWP - 1) / BLOCK_SIZE_M_LOWP;
        packedlhs_block_in_bytes = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(BLOCK_SIZE_M_LOWP, K, mr, kr, sr);
        lhsPackedSize = packedlhs_block_in_bytes * _m_blocks;
        if (KaiExecutorImpl == IMPL_TYPE::GatherMatmul) {
            const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
            const auto srcPrc = srcDesc->getPrecision();
            const auto dstPrc = dstDesc->getPrecision();
            m_tmpInputDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcPrc, Shape({M, K}));
            m_tmpOutputDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(dstPrc, Shape({M, N}));
            lhsPackedSize = rnd_up(lhsPackedSize, 64);
            auto srcSize = rnd_up(m_tmpInputDesc->getCurrentMemSize(), 64);  // 64 bytes is the cache line size
            auto dstSize = rnd_up(m_tmpOutputDesc->getCurrentMemSize(), 64);
            const size_t totalSize = lhsPackedSize + srcSize + dstSize;
            auto lhsPackedDesc = std::make_shared<CpuBlockedMemoryDesc>(i8, Shape({totalSize}));
            lhsPackedMem = scratchPad->createScratchPadMem(lhsPackedDesc);
        } else {
            auto lhsPackedDesc = std::make_shared<CpuBlockedMemoryDesc>(i8, Shape({lhsPackedSize}));
            lhsPackedMem = scratchPad->createScratchPadMem(lhsPackedDesc);
        }
    }
    return true;
}

void MatMulKleidiAIExecutor::execute(const MemoryArgs& memory) {
    const auto& cpu_parallel = executorContext->getCpuParallel();
    auto srcMem = memory.at(ARG_SRC);
    auto weiMem = memory.at(ARG_WEI);
    auto dstMem = memory.at(ARG_DST);
    auto srcDims = srcMem->getDesc().getShape().getDims();
    auto weiDims = weiMem->getDesc().getShape().getDims();
    size_t M = 0, K = 0, N = 0;
    if (KaiExecutorImpl == IMPL_TYPE::GatherMatmul) {
        // Gather rows that are processed by this expert.
        OPENVINO_ASSERT(!gather_idx.empty(), "gather_idx is not set");
        M = srcDims[1];
        K = srcDims[2];
        N = weiDims[0];
        const auto element_size = m_tmpInputDesc->getPrecision().size();
        auto* input_ptr = lhsPackedMem->getDataAs<uint8_t>() + lhsPackedSize;
        auto* output_ptr =
            input_ptr + rnd_up(m_tmpInputDesc->getCurrentMemSize(), 64);  // 64 bytes is the cache line size
        auto tmpInput = std::make_shared<Memory>(executorContext->getEngine(), m_tmpInputDesc, input_ptr);
        auto tmpOutput = std::make_shared<Memory>(executorContext->getEngine(), m_tmpOutputDesc, output_ptr);
        auto src_offset = OffsetHelper::createOffsetHelper(*srcMem);
        auto tmp_input_offset = OffsetHelper::createOffsetHelper(*tmpInput);
        cpu_parallel->parallel_for(gather_idx.size(), [&](size_t m) {
            auto* dst_row = tmp_input_offset(m);
            const auto row_id = gather_idx[m].first;
            const auto batch_index = gather_idx[m].second;
            const auto* src_data = src_offset(batch_index, row_id);
            std::memcpy(dst_row, src_data, K * element_size);
        });
        // update M
        M = gather_idx.size();
        srcMem = tmpInput;
        dstMem = tmpOutput;
    } else {
        srcDims = normalizeDimsTo2D(srcDims);
        M = srcDims[0];
        K = srcDims[1];
        N = weiDims[0];
    }
    const size_t lhs_stride = K * sizeof(float);
    const size_t dst_stride_row = N * sizeof(float);
    const size_t dst_stride_col = sizeof(float);
    auto* lhs = srcMem->getDataAs<float>();
    auto* dst = dstMem->getDataAs<float>();

    size_t n_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (!useDynamicQuant) {
        auto* rhs_packed = static_cast<float*>(rhsPackedMem->getData());

        cpu_parallel->parallel_for(n_blocks, [&](size_t n_block) {
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
        auto* lhs_packed_lowp = lhsPackedMem->getDataAs<int8_t>();
        auto* rhs_packed_lowp = rhsPackedMem->getDataAs<int8_t>();
        if (INT4_IMPL) {
            size_t m_step = BLOCK_SIZE_M_LOWP;
            size_t n_step = ukernel_i4->get_n_step();
            const size_t M_BLOCKS = (M + m_step - 1) / m_step;
            const size_t N_BLOCKS = (N + n_step - 1) / n_step;
            const size_t lhs_packed_offset = ukernel_i4->get_lhs_packed_offset(0, K);

            ParallelNestingContext nested_context;
            cpu_parallel->parallel_for(M_BLOCKS, [&](size_t m_blk) {
                const size_t M_iter = std::min(M - m_blk * m_step, m_step);
                auto* lhs_packed_block = lhs_packed_lowp + m_blk * packedlhs_block_in_bytes;

                kai_run_lhs_quant_pack_qai8dxp_f32(M_iter,
                                                   K,
                                                   mr,
                                                   kr,
                                                   sr,
                                                   0,
                                                   lhs + m_blk * m_step * K,  // LHS (F32)
                                                   lhs_stride,
                                                   lhs_packed_block  // lhs packed output
                );
                cpu_parallel->parallel_for(N_BLOCKS, [&](size_t n_blk) {
                    //  matmul exec
                    const size_t rhs_packed_offset = ukernel_i4->get_rhs_packed_offset(n_blk * n_step, K);
                    const size_t dst_offset =
                        ukernel_i4->get_dst_offset(m_blk * m_step, n_blk * n_step, dst_stride_row);
                    const auto* rhs_ptr = static_cast<const void*>(rhs_packed_lowp + rhs_packed_offset);
                    const auto* lhs_ptr = static_cast<const void*>(lhs_packed_block + lhs_packed_offset);
                    float* dst_ptr = (dst + dst_offset / sizeof(float));
                    const size_t N_iter = std::min(N - n_blk * n_step, n_step);
                    ukernel_i4->run_matmul(M_iter,
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
        } else {
            size_t m_step = BLOCK_SIZE_M_LOWP;
            constexpr size_t n_step = 4;
            const size_t M_BLOCKS = (M + m_step - 1) / m_step;
            const size_t N_BLOCKS = (N + n_step - 1) / n_step;
            const size_t lhs_packed_offset = ukernel_i8->get_lhs_packed_offset(0, K);

            ParallelNestingContext nested_context;
            cpu_parallel->parallel_for(M_BLOCKS, [&](size_t m_blk) {
                const size_t M_iter = std::min(M - m_blk * m_step, m_step);
                auto* lhs_packed_block = lhs_packed_lowp + m_blk * packedlhs_block_in_bytes;

                kai_run_lhs_quant_pack_qai8dxp_f32(M_iter,
                                                   K,
                                                   mr,
                                                   kr,
                                                   sr,
                                                   0,
                                                   lhs + m_blk * m_step * K,  // LHS (F32)
                                                   lhs_stride,
                                                   lhs_packed_block  // lhs packed output
                );
                cpu_parallel->parallel_for(N_BLOCKS, [&](size_t n_blk) {
                    //  matmul exec
                    const size_t rhs_packed_offset = ukernel_i8->get_rhs_packed_offset(n_blk * n_step, K);
                    const size_t dst_offset =
                        ukernel_i8->get_dst_offset(m_blk * m_step, n_blk * n_step, dst_stride_row);
                    const auto* rhs_ptr = static_cast<const void*>(rhs_packed_lowp + rhs_packed_offset);
                    const auto* lhs_ptr = static_cast<const void*>(lhs_packed_block + lhs_packed_offset);
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

    if (KaiExecutorImpl == IMPL_TYPE::GatherMatmul) {
        dstMem = memory.at(ARG_DST);
        const auto element_size = m_tmpInputDesc->getPrecision().size();
        auto* input_ptr = lhsPackedMem->getDataAs<uint8_t>() + lhsPackedSize;
        auto* output_ptr =
            input_ptr + rnd_up(m_tmpInputDesc->getCurrentMemSize(), 64);  // 64 bytes is the cache line size
        auto tmpInput = std::make_shared<Memory>(executorContext->getEngine(), m_tmpInputDesc, input_ptr);
        auto tmpOutput = std::make_shared<Memory>(executorContext->getEngine(), m_tmpOutputDesc, output_ptr);
        auto dst_offset = OffsetHelper::createOffsetHelper(dstMem);
        auto tmp_dst_offset = OffsetHelper::createOffsetHelper(tmpOutput);
        cpu_parallel->parallel_for(gather_idx.size(), [&](size_t m) {
            const auto* src_row = tmp_dst_offset(m);
            const auto row_id = gather_idx[m].first;
            const auto batch_index = gather_idx[m].second;
            auto* dst_row = dst_offset(batch_index, row_id);
            std::memcpy(dst_row, src_row, N * element_size);
        });
    }
}

void MatMulKleidiAIExecutor::moveMemToNumaNode([[maybe_unused]] int numaNodeID) {
    OPENVINO_THROW_NOT_IMPLEMENTED("'moveMemToNumaNode' is not implemented by the executor");
}

}  // namespace ov::intel_cpu
