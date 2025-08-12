// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak/xbyak.h>

#include <cassert>
#include <common/utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstdint>

#include "jit_kernel_base.hpp"
#include "nodes/kernels/x64/registers_pool.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

enum class GridSampleInterpolationMode : uint8_t { BILINEAR, BICUBIC, NEAREST };
enum class GridSamplePaddingMode : uint8_t { ZEROS, BORDER, REFLECTION };

namespace kernel {

class GridSampleKernelBase;

#if defined(OPENVINO_ARCH_X86_64)

struct GridSampleKernelConfParams {
    bool dynamicShapes = false;
    bool dynamicBatch = false;
    bool dynamicChannel = false;
    bool alignCorners = false;
    GridSampleInterpolationMode interpolationMode = GridSampleInterpolationMode::BILINEAR;
    GridSamplePaddingMode paddingMode = GridSamplePaddingMode::ZEROS;
    ov::element::Type inDataPrc;
    ov::element::Type gridPrc;
    uint64_t batchNum = 1LU;
    uint64_t cannelNum = 1LU;
    uint64_t srcBatchStepB = 0LU;
};

struct GridSamplesKernelExecArgs {
    const void* src = nullptr;
    const void* grid = nullptr;
    void* dst = nullptr;
    uint64_t batchNum = 1LU;
    uint64_t channelsNum = 1LU;
    const float* srcWidthF = nullptr;
    const float* srcHeightF = nullptr;
    uint64_t srcBatchStepB = 0LU;
    uint64_t gridBatchStepB = 0LU;
    uint64_t dstBatchStepB = 0LU;
    uint64_t srcChannelStepB = 0LU;
    uint64_t dstChannelStepB = 0LU;
    const void* wDenormCoefF = nullptr;
    const void* hDenormCoefF = nullptr;
    const void* srcWidthB = nullptr;
    const void* srcHeightMul2F = nullptr;
    const void* srcWidthMul2F = nullptr;
    const void* srcHeightMul2Sub1F = nullptr;
    const void* srcWidthMul2Sub1F = nullptr;
    const void* srcHeightSub1F = nullptr;
    const void* srcWidthSub1F = nullptr;
    const void* dataTypeSize = nullptr;
    const void* buffer = nullptr;
    uint64_t workAmount = 0LU;
};

enum coord : uint8_t { w, h };

class GridSampleKernelBase : public JitKernelBase {
public:
    void (*ker_)(const GridSamplesKernelExecArgs*) = nullptr;
    void operator()(const GridSamplesKernelExecArgs* args) const {
        assert(ker_);
        ker_(args);
    }
    explicit GridSampleKernelBase(const char* name,
                                  const GridSampleKernelConfParams& jcp,
                                  dnnl::impl::cpu::x64::cpu_isa_t isa,
                                  uint64_t vlen)
        : JitKernelBase(name, isa),

          jcp(jcp),
          vlen(vlen),
          dataTypeSize(jcp.inDataPrc.size()),
          gridTypeSize(jcp.gridPrc.size()),
          dataElPerVec(vlen / dataTypeSize),
          gridElPerVec(vlen / gridTypeSize) {}

    virtual void create_ker() = 0;
    uint64_t getVecLen() const {
        return vlen;
    }
    uint64_t getDataElPerVec() const {
        return dataElPerVec;
    }
    uint64_t getGridElPerVec() const {
        return gridElPerVec;
    }

protected:
    GridSampleKernelConfParams jcp;
    uint64_t vlen = 16LU;
    uint64_t dataTypeSize = 1LU;
    uint64_t gridTypeSize = 1LU;
    uint64_t dataElPerVec = 1LU;
    uint64_t gridElPerVec = 1LU;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
class GridSampleKernel : public GridSampleKernelBase {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(GridSampleKernel)

    explicit GridSampleKernel(const GridSampleKernelConfParams& jcp);

    void create_ker() override;
    void generate() override;

    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::avx512_core,
                                                         Xbyak::Zmm,
                                                         isa == dnnl::impl::cpu::x64::sse41,
                                                         Xbyak::Xmm,
                                                         Xbyak::Ymm>::type;
    using Vmask = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::avx512_core,
                                                           Xbyak::Opmask,
                                                           isa == dnnl::impl::cpu::x64::sse41,
                                                           Xbyak::Xmm,
                                                           Xbyak::Ymm>::type;

private:
    uint8_t dataTypeShift = 0;

    // Suffix "B" means "In Bytes", "F" - float.
    // 64b registers.
    RegistersPool::Reg<Xbyak::Reg64> regSrc;
    RegistersPool::Reg<Xbyak::Reg64> regGrid;
    RegistersPool::Reg<Xbyak::Reg64> regDst;
    RegistersPool::Reg<Xbyak::Reg64> regChannelNum;
    RegistersPool::Reg<Xbyak::Reg64> regWorkAmount;
    RegistersPool::Reg<Xbyak::Reg64> regSrcChannelStepB;
    RegistersPool::Reg<Xbyak::Reg64> regDstChannelStepB;

    const Xbyak::Reg64 regParams = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    // Tail mask.
    RegistersPool::Reg<Vmask> kTailMask;

    // Vector registers.
    RegistersPool::Reg<Vmm> vSrcHeightF;
    RegistersPool::Reg<Vmm> vSrcWidthF;
    RegistersPool::Reg<Vmm> vZeros;
    RegistersPool::Reg<Vmm> vHalfF;
    RegistersPool::Reg<Vmm> vOnesF;
    RegistersPool::Reg<Vmm> vWDenormCoefF;
    RegistersPool::Reg<Vmm> vHDenormCoefF;
    RegistersPool::Reg<Vmm> vGridPermMask;
    RegistersPool::Reg<Vmm> vDataTypeSizeB;  // for ZEROS padding
    RegistersPool::Reg<Vmm> vSrcWidthB;      // for ZEROS padding

    RegistersPool::Reg<Vmm> vSrcHeightSub1F;  // for BORDER padding
    RegistersPool::Reg<Vmm> vSrcWidthSub1F;   // for BORDER padding

    RegistersPool::Reg<Vmm> vSrcHeightMul2F;      // for REFLECTION padding
    RegistersPool::Reg<Vmm> vSrcWidthMul2F;       // for REFLECTION padding
    RegistersPool::Reg<Vmm> vSrcHeightMul2Sub1F;  // for REFLECTION padding
    RegistersPool::Reg<Vmm> vSrcWidthMul2Sub1F;   // for REFLECTION padding
    RegistersPool::Reg<Vmm> vAbsMask;             // for REFLECTION padding

    RegistersPool::Reg<Vmm> vConst_0_75;  // for BICUBIC interpolation
    RegistersPool::Reg<Vmm> vConst_1_25;  // for BICUBIC interpolation
    RegistersPool::Reg<Vmm> vConst_1_50;  // for BICUBIC interpolation
    RegistersPool::Reg<Vmm> vConst_2_00;  // for BICUBIC interpolation
    RegistersPool::Reg<Vmm> vConst_2_25;  // for BICUBIC interpolation

    void initVectors();
    void process();
    void spatialLoop();
    void getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord);
    void getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord);
    void denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord);
    void interpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail = false);
    void bilinearInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail = false);
    void bicubicInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail = false);
    void nearestInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail = false);
    void zerosPadding(const Vmask& kDst, const Vmm& vHCoord, const Vmm& vWCoord);
    void zerosPaddingW(const Vmask& kDst, const Vmm& vCoord);
    void zerosPaddingH(const Vmask& kDst, const Vmm& vCoord, const Vmask& kMaskW);
    void borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, coord dim);
    void reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, coord dim);
    void bicubicCoefficients(const Vmm& vCoef, const Vmm& vDX, uint8_t idx);
    void tail();

    // Aux
    void dataTypeShiftPs2Dq(const Vmm& vDst, const Vmm& vSrc);
    void hwShiftPs2dq(const Vmm& vDst, const Vmm& vHCoord, const Vmm& vWCoord, const Vmm& vWidth);
};

#endif  // OPENVINO_ARCH_X86_64

}  // namespace kernel
}  // namespace ov::intel_cpu
