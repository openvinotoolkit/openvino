// Copyright (C) 2026 FUJITSU LIMITED
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <variant>
#include <vector>

#include "arm_neon.h"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "kai/kai_common.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/executors/acl/acl_fullyconnected_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/debug_capabilities.h"
#include "utils/precision_support.h"

// Headers for F32 KAI kernels
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"

// Headers for INT8 KAI kernels
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp_qsi8cx_neon.h"

// Headers for INT4 KAI kernels
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp_qsi4cxp_interface.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"

// Headers for INT4 group symmetric KAI kernels
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h"

#define FLOAT_MAX 3.4028235e38f
#define FLOAT_MIN (-3.4028235e38f)

namespace ov::intel_cpu::kai_common {
enum class KAIKernelTag : std::uint8_t {
    F32_NEON_MLA,
    I8_NEON_DOTPROD,
    I8_NEON_IMM,
    I4_NEON_DOTPROD,
    I4_NEON_IMM,
    I4_NEON_IMM_GROUP,
    I4_NEON_DOTPROD_GROUP
};

using KernelInterface = std::variant<kai_matmul_clamp_f32_f32_f32p_ukernel,
                                     kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel,
                                     kai_matmul_clamp_f32_qai8dxp_qsi4cxp_ukernel,
                                     kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel>;

class uKernelBase {
protected:
    size_t M = 0UL, N = 0UL, K = 0UL;
    size_t BLOCK_SIZE = 0UL, sr = 0UL, kr = 0UL, mr = 0UL, nr = 0UL;
    size_t packedlhs_block_in_bytes = 0UL;

public:
    uKernelBase() = default;
    virtual ~uKernelBase() = default;

    /// @brief Returns the kernelTag associated with the object
    /// @return kai_common::KAIKernelTag
    virtual KAIKernelTag getKernelTag() = 0;

    /// @brief Return the micro-kernel interface associated with the object
    /// @return
    virtual KernelInterface getuKernelInterface() = 0;

    /// @brief Execute function to run matmul
    /// @param cpu_parallel
    /// @param M
    /// @param K
    /// @param dstMem
    /// @param srcMem
    virtual void execute(const ov::intel_cpu::CpuParallelPtr& cpu_parallel,
                         ov::intel_cpu::Dim M,
                         ov::intel_cpu::Dim K,
                         ov::intel_cpu::MemoryPtr dstMem,
                         ov::intel_cpu::MemoryPtr srcMem) = 0;

    /// @brief Returns the size required for RHS packing
    /// @return
    virtual size_t get_rhsPackedSize() {
        return 0;
    }

    /// @brief Pack the kernel data
    /// @param isTransposed [true/false] specify if tranposed or not
    /// @param weightsMemory MemoryPtr to weights
    /// @param biasMem
    /// @param hasBias
    /// @param rhs_scales
    /// @param rhsPackedMemory
    virtual void packData(bool isTransposed,
                          MemoryCPtr weightsMemory,
                          MemoryPtr biasMem,
                          bool hasBias,
                          float* rhs_scales,
                          MemoryPtr rhsPackedMemory) = 0;

    /// @brief Returns the size of LHS packing
    /// @param m
    /// @return
    virtual size_t getLHSPackedSize([[maybe_unused]] size_t m) {
        return 0;
    }
};

template <KAIKernelTag tag>
class uKernel;

// uKernel specialized for FP32
template <>
class uKernel<KAIKernelTag::F32_NEON_MLA> : public uKernelBase {
private:
    static constexpr kai_matmul_clamp_f32_f32_f32p_ukernel uKernelInterface{
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
    MemoryPtr rhsPackedMem;

public:
    uKernel(size_t N, size_t K) {
        this->N = N;
        this->K = K;
        this->BLOCK_SIZE = 8;
        this->nr = uKernelInterface.get_nr();
        this->kr = uKernelInterface.get_kr();
        this->sr = uKernelInterface.get_sr();
    }  // end of uKernel()...

    size_t get_rhsPackedSize() override {
        return kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(N, K);
    }

    void packData([[maybe_unused]] bool isTransposed,
                  MemoryCPtr packedWeights,
                  MemoryPtr biasMem,
                  [[maybe_unused]] bool hasBias,
                  [[maybe_unused]] float* rhs_scales,
                  MemoryPtr rhsPackedMemory) override {
        auto* rhs = static_cast<float*>(packedWeights->getData());
        rhsPackedMem = rhsPackedMemory;
        const size_t rhs_stride = N * sizeof(float);
        auto* bias = biasMem->getDataAs<float>();
        auto* rhs_packed = static_cast<float*>(rhsPackedMemory->getData());
        kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(1,
                                                         N,
                                                         K,
                                                         nr,
                                                         kr,
                                                         sr,
                                                         rhs_stride,
                                                         rhs,
                                                         bias,
                                                         nullptr,
                                                         rhs_packed,
                                                         0,
                                                         nullptr);
    }

    KernelInterface getuKernelInterface() override {
        return uKernelInterface;
    }

    KAIKernelTag getKernelTag() override {
        return KAIKernelTag::F32_NEON_MLA;
    }

    void execute(const ov::intel_cpu::CpuParallelPtr& cpu_parallel,
                 ov::intel_cpu::Dim M,
                 ov::intel_cpu::Dim K,
                 ov::intel_cpu::MemoryPtr dstMem,
                 ov::intel_cpu::MemoryPtr srcMem) override {
        auto _ukernel = std::get<kai_matmul_clamp_f32_f32_f32p_ukernel>(getuKernelInterface());
        size_t n_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        auto* rhs_packed = static_cast<float*>(rhsPackedMem->getData());
        const size_t dst_stride_row = N * sizeof(float);
        auto* dst = dstMem->getDataAs<float>();
        auto* lhs = srcMem->getDataAs<float>();
        const size_t lhs_stride = K * sizeof(float);
        const size_t dst_stride_col = sizeof(float);

        cpu_parallel->parallel_for(n_blocks, [&](size_t n_block) {
            size_t n_start = (n_block * BLOCK_SIZE);
            size_t n_end = std::min(n_start + BLOCK_SIZE, N);
            size_t n_block_size = n_end - n_start;
            const size_t rhs_packed_offset = _ukernel.get_rhs_packed_offset(n_start, K);
            const size_t dst_offset = _ukernel.get_dst_offset(0, n_start, dst_stride_row);
            const float* rhs_ptr = (rhs_packed + rhs_packed_offset / sizeof(float));
            float* dst_ptr = (dst + dst_offset / (sizeof(float)));

            _ukernel.run_matmul(M,
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
    }  // end of execute()...
};

template <>
class uKernel<KAIKernelTag::I8_NEON_DOTPROD> : public uKernelBase {
private:
    static constexpr kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel uKernelInterface{
        kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
        kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
        kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
        kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
        kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
        kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
        kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
        kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
        kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
        kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod,
        kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod};
    MemoryPtr rhsPackedMem;
    MemoryPtr& lhsPackedMem;

public:
    uKernel(size_t N, size_t K, MemoryPtr& lhsPackedMem) : lhsPackedMem(lhsPackedMem) {
        this->N = N;
        this->K = K;
        this->BLOCK_SIZE = 16;
        this->mr = uKernelInterface.get_mr();
        this->nr = uKernelInterface.get_nr();
        this->kr = uKernelInterface.get_kr();
        this->sr = uKernelInterface.get_sr();
    }  // end of uKernel()...

    size_t get_rhsPackedSize() override {
        return kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(N, K, nr, kr, sr);
    }

    void packData([[maybe_unused]] bool isTransposed,
                  MemoryCPtr weightsMemory,
                  MemoryPtr biasMem,
                  [[maybe_unused]] bool hasBias,
                  float* rhs_scales,
                  MemoryPtr rhsPackedMemory) override {
        rhsPackedMem = rhsPackedMemory;
        auto* rhs_native_qs8cx = weightsMemory->getDataAs<int8_t>();
        auto* bias = biasMem->getDataAs<float>();
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

    KernelInterface getuKernelInterface() override {
        return uKernelInterface;
    }

    KAIKernelTag getKernelTag() override {
        return KAIKernelTag::I8_NEON_DOTPROD;
    }

    size_t getLHSPackedSize(size_t m) override {
        const size_t _m_blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        packedlhs_block_in_bytes = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(BLOCK_SIZE, K, mr, kr, sr);
        const size_t lhsPackedSize = packedlhs_block_in_bytes * _m_blocks;
        return lhsPackedSize;
    }

    void execute(const ov::intel_cpu::CpuParallelPtr& cpu_parallel,
                 ov::intel_cpu::Dim M,
                 ov::intel_cpu::Dim K,
                 ov::intel_cpu::MemoryPtr dstMem,
                 ov::intel_cpu::MemoryPtr srcMem) override {
        auto _ukernel = std::get<kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel>(getuKernelInterface());
        size_t m_step = BLOCK_SIZE;
        constexpr size_t n_step = 4;
        const size_t M_BLOCKS = (M + m_step - 1) / m_step;
        const size_t N_BLOCKS = (N + n_step - 1) / n_step;
        const size_t lhs_packed_offset = _ukernel.get_lhs_packed_offset(0, K);
        const size_t lhs_stride = K * sizeof(float);

        auto* lhs = srcMem->getDataAs<float>();
        auto* dst = dstMem->getDataAs<float>();

        auto* lhs_packed_lowp = lhsPackedMem->getDataAs<int8_t>();
        auto* rhs_packed_lowp = rhsPackedMem->getDataAs<int8_t>();

        const size_t dst_stride_row = N * sizeof(float);
        const size_t dst_stride_col = sizeof(float);

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
                const size_t rhs_packed_offset = _ukernel.get_rhs_packed_offset(n_blk * n_step, K);
                const size_t dst_offset = _ukernel.get_dst_offset(m_blk * m_step, n_blk * n_step, dst_stride_row);
                const auto* rhs_ptr = static_cast<const void*>(rhs_packed_lowp + rhs_packed_offset);
                const auto* lhs_ptr = static_cast<const void*>(lhs_packed_block + lhs_packed_offset);
                float* dst_ptr = (dst + dst_offset / sizeof(float));
                const size_t N_iter = std::min(N - n_blk * n_step, n_step);
                _ukernel.run_matmul(M_iter,
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
    }  // end of execute()...
};

template <>
class uKernel<KAIKernelTag::I8_NEON_IMM> : public uKernelBase {
private:
    static constexpr kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel uKernelInterface{
        kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
        kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
        kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
        kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
        kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
        kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
        kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
        kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
        kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
        kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
        kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm};
    MemoryPtr rhsPackedMem;
    MemoryPtr& lhsPackedMem;

public:
    uKernel(size_t N, size_t K, MemoryPtr& lhsPackedMem) : lhsPackedMem(lhsPackedMem) {
        this->N = N;
        this->K = K;
        this->BLOCK_SIZE = 16;
        this->mr = uKernelInterface.get_mr();
        this->nr = uKernelInterface.get_nr();
        this->kr = uKernelInterface.get_kr();
        this->sr = uKernelInterface.get_sr();
    }  // end of uKernel()...

    size_t get_rhsPackedSize() override {
        return kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(N, K, nr, kr, sr);
    }

    void packData([[maybe_unused]] bool isTransposed,
                  MemoryCPtr weightsMemory,
                  MemoryPtr biasMem,
                  [[maybe_unused]] bool hasBias,
                  float* rhs_scales,
                  MemoryPtr rhsPackedMemory) override {
        rhsPackedMem = rhsPackedMemory;
        auto* rhs_native_qs8cx = weightsMemory->getDataAs<int8_t>();
        auto* bias = biasMem->getDataAs<float>();
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

    KernelInterface getuKernelInterface() override {
        return uKernelInterface;
    }

    KAIKernelTag getKernelTag() override {
        return KAIKernelTag::I8_NEON_IMM;
    }

    size_t getLHSPackedSize(size_t m) override {
        const size_t _m_blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        packedlhs_block_in_bytes = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(BLOCK_SIZE, K, mr, kr, sr);
        const size_t lhsPackedSize = packedlhs_block_in_bytes * _m_blocks;
        return lhsPackedSize;
    }

    void execute(const ov::intel_cpu::CpuParallelPtr& cpu_parallel,
                 ov::intel_cpu::Dim M,
                 ov::intel_cpu::Dim K,
                 ov::intel_cpu::MemoryPtr dstMem,
                 ov::intel_cpu::MemoryPtr srcMem) override {
        auto _ukernel = std::get<kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel>(getuKernelInterface());
        size_t m_step = BLOCK_SIZE;
        constexpr size_t n_step = 4;
        const size_t M_BLOCKS = (M + m_step - 1) / m_step;
        const size_t N_BLOCKS = (N + n_step - 1) / n_step;
        const size_t lhs_packed_offset = _ukernel.get_lhs_packed_offset(0, K);
        const size_t lhs_stride = K * sizeof(float);

        auto* lhs = srcMem->getDataAs<float>();
        auto* dst = dstMem->getDataAs<float>();

        auto* lhs_packed_lowp = lhsPackedMem->getDataAs<int8_t>();
        auto* rhs_packed_lowp = rhsPackedMem->getDataAs<int8_t>();

        const size_t dst_stride_row = N * sizeof(float);
        const size_t dst_stride_col = sizeof(float);

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
                const size_t rhs_packed_offset = _ukernel.get_rhs_packed_offset(n_blk * n_step, K);
                const size_t dst_offset = _ukernel.get_dst_offset(m_blk * m_step, n_blk * n_step, dst_stride_row);
                const auto* rhs_ptr = static_cast<const void*>(rhs_packed_lowp + rhs_packed_offset);
                const auto* lhs_ptr = static_cast<const void*>(lhs_packed_block + lhs_packed_offset);
                float* dst_ptr = (dst + dst_offset / sizeof(float));
                const size_t N_iter = std::min(N - n_blk * n_step, n_step);
                _ukernel.run_matmul(M_iter,
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
    }  // end of execute()...
};

template <>
class uKernel<KAIKernelTag::I4_NEON_DOTPROD> : public uKernelBase {
private:
    static constexpr kai_matmul_clamp_f32_qai8dxp_qsi4cxp_ukernel uKernelInterface{
        kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
        kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
        kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
        kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
        kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
        kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
        kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
        kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
        kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
        kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
        kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod};
    MemoryPtr rhsPackedMem;
    MemoryPtr& lhsPackedMem;

public:
    uKernel(size_t N, size_t K, MemoryPtr& lhsPackedMem) : lhsPackedMem(lhsPackedMem) {
        this->N = N;
        this->K = K;
        this->BLOCK_SIZE = uKernelInterface.get_m_step();
        this->mr = uKernelInterface.get_mr();
        this->nr = uKernelInterface.get_nr();
        this->kr = uKernelInterface.get_kr();
        this->sr = uKernelInterface.get_sr();
    }  // end of uKernel()...

    size_t get_rhsPackedSize() override {
        return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(N, K, nr, kr, sr);
    }

    void packData(bool isTransposed,
                  MemoryCPtr weightsMemory,
                  MemoryPtr biasMem,
                  bool hasBias,
                  float* rhs_scales,
                  MemoryPtr rhsPackedMemory) override {
        rhsPackedMem = rhsPackedMemory;
        kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params params{};
        params.lhs_zero_point = 1;
        params.rhs_zero_point = 0;
        auto* rhs_native_qs4cx = weightsMemory->getDataAs<uint8_t>();
        auto* bias_ptr = (hasBias) ? biasMem->getDataAs<float>() : nullptr;
        auto* rhs_packed_qs4cx = static_cast<int8_t*>(rhsPackedMem->getData());
        if (isTransposed) {
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
    }

    KernelInterface getuKernelInterface() override {
        return uKernelInterface;
    }

    KAIKernelTag getKernelTag() override {
        return KAIKernelTag::I4_NEON_DOTPROD;
    }

    size_t getLHSPackedSize(size_t m) override {
        const size_t _m_blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        this->packedlhs_block_in_bytes = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(BLOCK_SIZE, K, mr, kr, sr);
        const size_t lhsPackedSize = packedlhs_block_in_bytes * _m_blocks;
        return lhsPackedSize;
    }

    void execute(const ov::intel_cpu::CpuParallelPtr& cpu_parallel,
                 ov::intel_cpu::Dim M,
                 ov::intel_cpu::Dim K,
                 ov::intel_cpu::MemoryPtr dstMem,
                 ov::intel_cpu::MemoryPtr srcMem) override {
        kai_matmul_clamp_f32_qai8dxp_qsi4cxp_ukernel _ukernel =
            std::get<kai_matmul_clamp_f32_qai8dxp_qsi4cxp_ukernel>(getuKernelInterface());
        auto* lhs_packed_lowp = lhsPackedMem->getDataAs<int8_t>();
        auto* rhs_packed_lowp = rhsPackedMem->getDataAs<int8_t>();
        size_t m_step = _ukernel.get_m_step();
        size_t n_step = _ukernel.get_n_step();
        const size_t M_BLOCKS = (M + m_step - 1) / m_step;
        const size_t N_BLOCKS = (N + n_step - 1) / n_step;
        const size_t lhs_packed_offset = _ukernel.get_lhs_packed_offset(0, K);
        auto* lhs = srcMem->getDataAs<float>();
        auto* dst = dstMem->getDataAs<float>();
        const size_t lhs_stride = K * sizeof(float);
        const size_t dst_stride_row = N * sizeof(float);
        const size_t dst_stride_col = sizeof(float);

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
                const size_t rhs_packed_offset = _ukernel.get_rhs_packed_offset(n_blk * n_step, K);
                const size_t dst_offset = _ukernel.get_dst_offset(m_blk * m_step, n_blk * n_step, dst_stride_row);
                const auto* rhs_ptr = static_cast<const void*>(rhs_packed_lowp + rhs_packed_offset);
                const auto* lhs_ptr = static_cast<const void*>(lhs_packed_block + lhs_packed_offset);
                float* dst_ptr = (dst + dst_offset / sizeof(float));
                const size_t N_iter = std::min(N - n_blk * n_step, n_step);
                _ukernel.run_matmul(M_iter,
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

    }  // end of execute()...
};

template <>
class uKernel<KAIKernelTag::I4_NEON_IMM> : public uKernelBase {
private:
    static constexpr kai_matmul_clamp_f32_qai8dxp_qsi4cxp_ukernel uKernelInterface{
        kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
        kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm};
    MemoryPtr rhsPackedMem;
    MemoryPtr& lhsPackedMem;

public:
    uKernel(size_t N, size_t K, MemoryPtr& lhsPackedMem) : lhsPackedMem(lhsPackedMem) {
        this->N = N;
        this->K = K;
        this->BLOCK_SIZE = uKernelInterface.get_m_step();
        this->mr = uKernelInterface.get_mr();
        this->nr = uKernelInterface.get_nr();
        this->kr = uKernelInterface.get_kr();
        this->sr = uKernelInterface.get_sr();
    }  // end of uKernel()...

    size_t get_rhsPackedSize() override {
        return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(N, K, nr, kr, sr);
    }

    void packData(bool isTransposed,
                  MemoryCPtr weightsMemory,
                  MemoryPtr biasMem,
                  bool hasBias,
                  float* rhs_scales,
                  MemoryPtr rhsPackedMemory) override {
        rhsPackedMem = rhsPackedMemory;
        kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params params{};
        params.lhs_zero_point = 1;
        params.rhs_zero_point = 0;
        auto* rhs_native_qs4cx = weightsMemory->getDataAs<uint8_t>();
        auto* bias_ptr = (hasBias) ? biasMem->getDataAs<float>() : nullptr;
        auto* rhs_packed_qs4cx = static_cast<int8_t*>(rhsPackedMem->getData());
        if (isTransposed) {
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
    }

    KernelInterface getuKernelInterface() override {
        return uKernelInterface;
    }

    KAIKernelTag getKernelTag() override {
        return KAIKernelTag::I4_NEON_IMM;
    }

    size_t getLHSPackedSize(size_t m) override {
        const size_t _m_blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        this->packedlhs_block_in_bytes = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(BLOCK_SIZE, K, mr, kr, sr);
        const size_t lhsPackedSize = packedlhs_block_in_bytes * _m_blocks;
        return lhsPackedSize;
    }

    void execute(const ov::intel_cpu::CpuParallelPtr& cpu_parallel,
                 ov::intel_cpu::Dim M,
                 ov::intel_cpu::Dim K,
                 ov::intel_cpu::MemoryPtr dstMem,
                 ov::intel_cpu::MemoryPtr srcMem) override {
        kai_matmul_clamp_f32_qai8dxp_qsi4cxp_ukernel _ukernel =
            std::get<kai_matmul_clamp_f32_qai8dxp_qsi4cxp_ukernel>(getuKernelInterface());
        auto* lhs_packed_lowp = lhsPackedMem->getDataAs<int8_t>();
        auto* rhs_packed_lowp = rhsPackedMem->getDataAs<int8_t>();
        size_t m_step = _ukernel.get_m_step();
        size_t n_step = _ukernel.get_n_step();
        const size_t M_BLOCKS = (M + m_step - 1) / m_step;
        const size_t N_BLOCKS = (N + n_step - 1) / n_step;
        const size_t lhs_packed_offset = _ukernel.get_lhs_packed_offset(0, K);
        auto* lhs = srcMem->getDataAs<float>();
        auto* dst = dstMem->getDataAs<float>();
        const size_t lhs_stride = K * sizeof(float);
        const size_t dst_stride_row = N * sizeof(float);
        const size_t dst_stride_col = sizeof(float);

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
                const size_t rhs_packed_offset = _ukernel.get_rhs_packed_offset(n_blk * n_step, K);
                const size_t dst_offset = _ukernel.get_dst_offset(m_blk * m_step, n_blk * n_step, dst_stride_row);
                const auto* rhs_ptr = static_cast<const void*>(rhs_packed_lowp + rhs_packed_offset);
                const auto* lhs_ptr = static_cast<const void*>(lhs_packed_block + lhs_packed_offset);
                float* dst_ptr = (dst + dst_offset / sizeof(float));
                const size_t N_iter = std::min(N - n_blk * n_step, n_step);
                _ukernel.run_matmul(M_iter,
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

    }  // end of execute()...
};

template <>
class uKernel<KAIKernelTag::I4_NEON_IMM_GROUP> : public uKernelBase {
private:
    static constexpr kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel uKernelInterface{
        kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
        kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm};
    MemoryPtr rhsPackedMem;
    MemoryPtr& lhsPackedMem;
    size_t group_size;

public:
    uKernel(size_t N, size_t K, MemoryPtr& lhsPackedMem, const MemoryArgs& memory) : lhsPackedMem(lhsPackedMem) {
        this->N = N;
        this->K = K;
        auto scales = memory.at(ARG_WEI | ARG_ATTR_SCALES)->getDesc().getShape().getDims();
        OPENVINO_ASSERT(scales.size() > 1,
                        "Group quantization requires the scales tensor to have at least 2 dimensions.Got ",
                        scales.size(),
                        " dimension(s).");
        group_size = K / scales[1];
        OPENVINO_ASSERT(group_size % 32 == 0, "Group_size must be a multiple of 32");
        this->BLOCK_SIZE = uKernelInterface.get_m_step();
        this->mr = uKernelInterface.get_mr();
        this->nr = uKernelInterface.get_nr();
        this->kr = uKernelInterface.get_kr();
        this->sr = uKernelInterface.get_sr();
    }

    size_t get_rhsPackedSize() override {
        return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(N, K, nr, kr, sr, group_size, kai_dt_bf16);
    }

    void packData(bool isTransposed,
                  MemoryCPtr weightsMemory,
                  MemoryPtr biasMem,
                  bool hasBias,
                  float* rhs_scales,
                  MemoryPtr rhsPackedMemory) override {
        rhsPackedMem = rhsPackedMemory;
        auto* rhs_native_qs4cx = weightsMemory->getDataAs<uint8_t>();
        auto* bias_ptr = (hasBias) ? biasMem->getDataAs<float>() : nullptr;
        auto scale_stride = (K / group_size) * sizeof(uint16_t);
        auto* rhs_packed = rhsPackedMem->getData();

        // KAI packing kernel qsu4c32s1s0 - expects unsigned 4bit values, OpenVINO generates signed 2's complement,
        // hence the conversion is requied
        const size_t weightBytes = N * K / 2;
        std::vector<uint8_t> convertedRhs(weightBytes);
        for (size_t i = 0; i < weightBytes; i++) {
            convertedRhs[i] = rhs_native_qs4cx[i] ^ 0x88U;
        }

        // Convert F32 scales to bf16 as KAI kernel accepts only BF16
        const size_t numScales = N * (K / group_size);
        std::vector<uint16_t> scalesBf16(numScales);
        for (size_t i = 0; i < numScales; i++) {
            scalesBf16[i] = ov::bfloat16(rhs_scales[i]).to_bits();
        }

        if (isTransposed) {
            kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0_params params_kxn{};
            params_kxn.lhs_zero_point = 1;
            params_kxn.rhs_zero_point = 8;
            params_kxn.scale_dt = kai_dt_bf16;
            kai_run_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(1,
                                                      N,
                                                      K,
                                                      nr,
                                                      kr,
                                                      sr,
                                                      group_size,
                                                      convertedRhs.data(),
                                                      N / 2,
                                                      bias_ptr,
                                                      scalesBf16.data(),
                                                      scale_stride,
                                                      rhs_packed,
                                                      0,
                                                      &params_kxn);
        } else {
            kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params_nxk{};
            params_nxk.lhs_zero_point = 1;
            params_nxk.rhs_zero_point = 8;
            params_nxk.scale_dt = kai_dt_bf16;
            kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(1,
                                                      N,
                                                      K,
                                                      nr,
                                                      kr,
                                                      sr,
                                                      group_size,
                                                      convertedRhs.data(),
                                                      K / 2,
                                                      bias_ptr,
                                                      scalesBf16.data(),
                                                      scale_stride,
                                                      rhs_packed,
                                                      0,
                                                      &params_nxk);
        }
    }

    KernelInterface getuKernelInterface() override {
        return uKernelInterface;
    }

    KAIKernelTag getKernelTag() override {
        return KAIKernelTag::I4_NEON_IMM_GROUP;
    }

    size_t getLHSPackedSize(size_t m) override {
        const size_t _m_blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        this->packedlhs_block_in_bytes = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(BLOCK_SIZE, K, mr, kr, sr);
        const size_t lhsPackedSize = packedlhs_block_in_bytes * _m_blocks;
        return lhsPackedSize;
    }

    void execute(const ov::intel_cpu::CpuParallelPtr& cpu_parallel,
                 ov::intel_cpu::Dim M,
                 ov::intel_cpu::Dim K,
                 ov::intel_cpu::MemoryPtr dstMem,
                 ov::intel_cpu::MemoryPtr srcMem) override {
        kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel _ukernel =
            std::get<kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel>(getuKernelInterface());

        auto* lhs_packed_lowp = lhsPackedMem->getDataAs<int8_t>();
        auto* rhs_packed_lowp = rhsPackedMem->getDataAs<int8_t>();
        const size_t lhs_stride = K * sizeof(float);
        const size_t dst_stride_row = N * sizeof(float);
        const size_t dst_stride_col = sizeof(float);
        auto* lhs = srcMem->getDataAs<float>();
        auto* dst = dstMem->getDataAs<float>();
        size_t m_step = BLOCK_SIZE;
        size_t n_step = _ukernel.get_n_step();
        const size_t M_BLOCKS = (M + m_step - 1) / m_step;
        const size_t N_BLOCKS = (N + n_step - 1) / n_step;
        const size_t lhs_packed_offset = _ukernel.get_lhs_packed_offset(0, K);

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
                const size_t rhs_packed_offset = _ukernel.get_rhs_packed_offset(n_blk * n_step, K, group_size);
                const size_t dst_offset = _ukernel.get_dst_offset(m_blk * m_step, n_blk * n_step, dst_stride_row);
                const auto* rhs_ptr = static_cast<const void*>(rhs_packed_lowp + rhs_packed_offset);
                const auto* lhs_ptr = static_cast<const void*>(lhs_packed_block + lhs_packed_offset);
                float* dst_ptr = (dst + dst_offset / sizeof(float));
                const size_t N_iter = std::min(N - n_blk * n_step, n_step);

                _ukernel.run_matmul(M_iter,
                                    N_iter,
                                    K,
                                    group_size,
                                    lhs_ptr,
                                    rhs_ptr,
                                    dst_ptr,
                                    dst_stride_row,
                                    dst_stride_col,
                                    FLOAT_MIN,
                                    FLOAT_MAX);
            });
        });
    }  // end of execute()...
};

template <>
class uKernel<KAIKernelTag::I4_NEON_DOTPROD_GROUP> : public uKernelBase {
private:
    static constexpr kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel uKernelInterface{
        kai_get_m_step_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod,
        kai_get_n_step_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod,
        kai_get_mr_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod,
        kai_get_nr_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod,
        kai_get_kr_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod,
        kai_get_sr_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod,
        kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod,
        kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod,
        kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod,
        kai_get_dst_size_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod,
        kai_run_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod};
    MemoryPtr rhsPackedMem;
    MemoryPtr& lhsPackedMem;
    size_t group_size;

public:
    uKernel(size_t N, size_t K, MemoryPtr& lhsPackedMem, const MemoryArgs& memory) : lhsPackedMem(lhsPackedMem) {
        this->N = N;
        this->K = K;
        auto scales = memory.at(ARG_WEI | ARG_ATTR_SCALES)->getDesc().getShape().getDims();
        OPENVINO_ASSERT(scales.size() > 1,
                        "Group quantization requires the scales tensor to have at least 2 dimensions. Got ",
                        scales.size(),
                        " dimension(s).");
        group_size = K / scales[1];
        OPENVINO_ASSERT(group_size % 32 == 0, "Group_size must be a multiple of 32");
        this->BLOCK_SIZE = uKernelInterface.get_m_step();
        this->mr = uKernelInterface.get_mr();
        this->nr = uKernelInterface.get_nr();
        this->kr = uKernelInterface.get_kr();
        this->sr = uKernelInterface.get_sr();
    }

    size_t get_rhsPackedSize() override {
        return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(N, K, nr, kr, sr, group_size, kai_dt_bf16);
    }

    void packData(bool isTransposed,
                  MemoryCPtr weightsMemory,
                  MemoryPtr biasMem,
                  bool hasBias,
                  float* rhs_scales,
                  MemoryPtr rhsPackedMemory) override {
        rhsPackedMem = rhsPackedMemory;
        auto* rhs_native_qs4cx = weightsMemory->getDataAs<uint8_t>();
        auto* bias_ptr = (hasBias) ? biasMem->getDataAs<float>() : nullptr;
        auto scale_stride = (K / group_size) * sizeof(uint16_t);
        auto* rhs_packed = rhsPackedMem->getData();

        // KAI packing kernel qsu4c32s1s0 - expects unsigned 4bit values, OpenVINO generates signed 2's complement,
        // hence the conversion is requied
        const size_t weightBytes = N * K / 2;
        std::vector<uint8_t> convertedRhs(weightBytes);
        for (size_t i = 0; i < weightBytes; i++) {
            convertedRhs[i] = rhs_native_qs4cx[i] ^ 0x88U;
        }

        // Convert F32 scales to bf16 as KAI kernel accepts only BF16
        const size_t numScales = N * (K / group_size);
        std::vector<uint16_t> scalesBf16(numScales);
        for (size_t i = 0; i < numScales; i++) {
            scalesBf16[i] = ov::bfloat16(rhs_scales[i]).to_bits();
        }

        if (isTransposed) {
            kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0_params params_kxn{};
            params_kxn.lhs_zero_point = 1;
            params_kxn.rhs_zero_point = 8;
            params_kxn.scale_dt = kai_dt_bf16;
            kai_run_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(1,
                                                      N,
                                                      K,
                                                      nr,
                                                      kr,
                                                      sr,
                                                      group_size,
                                                      convertedRhs.data(),
                                                      N / 2,
                                                      bias_ptr,
                                                      scalesBf16.data(),
                                                      scale_stride,
                                                      rhs_packed,
                                                      0,
                                                      &params_kxn);
        } else {
            kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params_nxk{};
            params_nxk.lhs_zero_point = 1;
            params_nxk.rhs_zero_point = 8;
            params_nxk.scale_dt = kai_dt_bf16;
            kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(1,
                                                      N,
                                                      K,
                                                      nr,
                                                      kr,
                                                      sr,
                                                      group_size,
                                                      convertedRhs.data(),
                                                      K / 2,
                                                      bias_ptr,
                                                      scalesBf16.data(),
                                                      scale_stride,
                                                      rhs_packed,
                                                      0,
                                                      &params_nxk);
        }
    }

    KernelInterface getuKernelInterface() override {
        return uKernelInterface;
    }

    KAIKernelTag getKernelTag() override {
        return KAIKernelTag::I4_NEON_DOTPROD_GROUP;
    }

    size_t getLHSPackedSize(size_t m) override {
        const size_t _m_blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        this->packedlhs_block_in_bytes = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(BLOCK_SIZE, K, mr, kr, sr);
        const size_t lhsPackedSize = packedlhs_block_in_bytes * _m_blocks;
        return lhsPackedSize;
    }

    void execute(const ov::intel_cpu::CpuParallelPtr& cpu_parallel,
                 ov::intel_cpu::Dim M,
                 ov::intel_cpu::Dim K,
                 ov::intel_cpu::MemoryPtr dstMem,
                 ov::intel_cpu::MemoryPtr srcMem) override {
        kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel _ukernel =
            std::get<kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel>(getuKernelInterface());

        auto* lhs_packed_lowp = lhsPackedMem->getDataAs<int8_t>();
        auto* rhs_packed_lowp = rhsPackedMem->getDataAs<int8_t>();
        const size_t lhs_stride = K * sizeof(float);
        const size_t dst_stride_row = N * sizeof(float);
        const size_t dst_stride_col = sizeof(float);
        auto* lhs = srcMem->getDataAs<float>();
        auto* dst = dstMem->getDataAs<float>();
        size_t m_step = BLOCK_SIZE;
        size_t n_step = _ukernel.get_n_step();
        const size_t M_BLOCKS = (M + m_step - 1) / m_step;
        const size_t N_BLOCKS = (N + n_step - 1) / n_step;
        const size_t lhs_packed_offset = _ukernel.get_lhs_packed_offset(0, K);

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
                const size_t rhs_packed_offset = _ukernel.get_rhs_packed_offset(n_blk * n_step, K, group_size);
                const size_t dst_offset = _ukernel.get_dst_offset(m_blk * m_step, n_blk * n_step, dst_stride_row);
                const auto* rhs_ptr = static_cast<const void*>(rhs_packed_lowp + rhs_packed_offset);
                const auto* lhs_ptr = static_cast<const void*>(lhs_packed_block + lhs_packed_offset);
                float* dst_ptr = (dst + dst_offset / sizeof(float));
                const size_t N_iter = std::min(N - n_blk * n_step, n_step);

                _ukernel.run_matmul(M_iter,
                                    N_iter,
                                    K,
                                    group_size,
                                    lhs_ptr,
                                    rhs_ptr,
                                    dst_ptr,
                                    dst_stride_row,
                                    dst_stride_col,
                                    FLOAT_MIN,
                                    FLOAT_MAX);
            });
        });
    }  // end of execute()...
};
}  // namespace ov::intel_cpu::kai_common