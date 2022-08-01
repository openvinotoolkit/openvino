// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grid_sample_kernel.hpp"
#include <ie_common.h>

using namespace dnnl::impl::cpu;

namespace ov {
namespace intel_cpu {

template <>
const unsigned jitGridSampleKernel<x64::avx512_core>::gridPermMask[16]  = { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 };
template <>
const unsigned jitGridSampleKernel<x64::sse41>::gridPermMask[1]  = { 0 };
template <>
const unsigned jitGridSampleKernel<x64::avx2>::gridPermMask[8]  = { 0, 2, 4, 6, 1, 3, 5, 7 };
template <>
const unsigned jitGridSampleKernel<x64::avx>::gridPermMask[8]  = { 0, 2, 4, 6, 1, 3, 5, 7 };

template <>
const float jitGridSampleKernel<x64::sse41>::halfValuesF[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
template <x64::cpu_isa_t isa>
const float jitGridSampleKernel<isa>::halfValuesF[1] = { 0.5f };

template <>
const float jitGridSampleKernel<x64::sse41>::oneValuesF[4] = { 1.f, 1.f, 1.f, 1.f };
template <x64::cpu_isa_t isa>
const float jitGridSampleKernel<isa>::oneValuesF[1] = { 1.f };

template <>
const unsigned jitGridSampleKernel<x64::avx512_core>::absMask[1] = { 0x7fffffff };
template <>
const unsigned jitGridSampleKernel<x64::sse41>::absMask[4] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
template <x64::cpu_isa_t isa>
const unsigned jitGridSampleKernel<isa>::absMask[8] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };

template <>
float jitGridSampleKernel<x64::avx>::dataTypeSizeArr[8];
template <x64::cpu_isa_t isa>
float jitGridSampleKernel<isa>::dataTypeSizeArr[1];

#define GET_OFF(field) offsetof(jGridSamplesExecArgs, field)

template <x64::cpu_isa_t isa>
jitGridSampleKernel<isa>::jitGridSampleKernel(const jGridSampleConfParams& jcp) :
        jitGridSampleKernelBase(jcp) {
    vlen = x64::cpu_isa_traits<isa>::vlen;
    dataTypeSize = jcp.inDataPrc.size();
    gridTypeSize = jcp.gridPrc.size();
    dataElPerVec = vlen / dataTypeSize;
    gridElPerVec = vlen / gridTypeSize;
    if (dataTypeSize == 2)
        dataTypeShift = 1;
    else if (dataTypeSize == 4)
        dataTypeShift = 2;
    if (isa == x64::avx) {
        for (int i = 0; i < dataElPerVec; i++) {
            dataTypeSizeArr[i] = dataTypeSize;
        }
    }
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::create_ker() {
    auto code = x64::jit_generator::create_kernel();
    if (code != dnnl::impl::status::success)
        IE_THROW() << "Could not create GridSample kernel. Error code: " << std::to_string(code);
    ker_ = (decltype(ker_))jit_ker();
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::generate() {
    this->preamble();

    mov(regSrc,  ptr[regParams + GET_OFF(src)]);
    mov(regDst,  ptr[regParams + GET_OFF(dst)]);
    mov(regGrid, ptr[regParams + GET_OFF(grid)]);

    mov(regAux1, ptr[regParams + GET_OFF(srcWidthF)]);
    uni_vpbroadcastd(vSrcWidthF, ptr[regAux1]);
    mov(regAux1, ptr[regParams + GET_OFF(srcHeightF)]);
    uni_vpbroadcastd(vSrcHeightF, ptr[regAux1]);

    mov(regSrcChannelStepB, ptr[regParams + GET_OFF(srcChannelStepB)]);
    mov(regDstChannelStepB, ptr[regParams + GET_OFF(dstChannelStepB)]);

    uni_vpxor(vZeros, vZeros, vZeros);

    if (isa == x64::avx512_core || isa == x64::avx2 || isa == x64::avx) {
        mov(regChannelsNum, ptr[regParams + GET_OFF(channelsNum)]);

        mov(regAux1, reinterpret_cast<uintptr_t>(oneValuesF));
        vbroadcastss(vOnesF, ptr[regAux1]);

        if (jcp.alignCorners) {
            mov(regAux1, ptr[regParams + GET_OFF(wDenormCoefF)]);
            vbroadcastss(vWDenormCoefF, ptr[regAux1]);
            mov(regAux1, ptr[regParams + GET_OFF(hDenormCoefF)]);
            vbroadcastss(vHDenormCoefF, ptr[regAux1]);
        } else {
            mov(regAux1, reinterpret_cast<uintptr_t>(halfValuesF));
            vbroadcastss(vHalfF, ptr[regAux1]);
        }

        mov(regAux1, reinterpret_cast<uintptr_t>(gridPermMask));
        uni_vmovups(vPermGridMask, ptr[regAux1]);

        if (isa == x64::avx512_core) {
            if (jcp.paddingMode == PaddingMode::ZEROS) {
                mov(regAux1, dataTypeSize);
                vpbroadcastd(vDataTypeSize, reg32Aux1);
                mov(regAux1, ptr[regParams + GET_OFF(srcWidthB)]);
                uni_vpbroadcastd(vSrcWidthB, ptr[regAux1]);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                mov(regAux1, ptr[regParams + GET_OFF(srcHeightSub1F)]);
                uni_vpbroadcastd(vSrcHeightSub1F, ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(srcWidthSub1F)]);
                uni_vpbroadcastd(vSrcWidthSub1F, ptr[regAux1]);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                mov(regAux1, ptr[regParams + GET_OFF(srcHeightMul2F)]);
                uni_vpbroadcastd(vSrcHeightMul2F, ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(srcWidthMul2F)]);
                uni_vpbroadcastd(vSrcWidthMul2F, ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
                uni_vpbroadcastd(vSrcHeightMul2Sub1F, ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
                uni_vpbroadcastd(vSrcWidthMul2Sub1F, ptr[regAux1]);
                if (jcp.alignCorners) {
                    mov(reg32Aux1, 0x7fffffff);
                    vpbroadcastd(vAbsMask, reg32Aux1);
                }
            }

            if (jcp.interpolationMode == InterpolationMode::BICUBIC) {
                mov(reg32Aux1, 0xbf400000); // -0.75f
                vpbroadcastd(vBicubConst, reg32Aux1);
                mov(reg32Aux1, 0x3fa00000); // 1.25f
                vpbroadcastd(vBicub2Const, reg32Aux1);
                mov(reg32Aux1, 0x40100000); // 2.25f
                vpbroadcastd(vBicub3Const, reg32Aux1);
                mov(reg32Aux1, 0x40000000); // 2.f
                vpbroadcastd(v2val, reg32Aux1); // TODO: rename
                mov(reg32Aux1, 0x3fc00000); // 1.5f
                vpbroadcastd(v2Bicub3Const, reg32Aux1);
            }
        }
    } else if (isa == x64::sse41) {
        mov(regAux1, reinterpret_cast<uintptr_t>(oneValuesF));
        uni_vmovups(vOnesF, ptr[regAux1]);
    }

    process();

    this->postamble();
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::process() {
    if (jcp.dynamicShapes) {
        Xbyak::Label lBatchLoop, lEnd;
        mov(regBatch, ptr[regParams + GET_OFF(batchNum)]);
        L(lBatchLoop);
        {
            cmp(regBatch, 0);
            jle(lEnd, T_NEAR);

            mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);
            spatialLoop(vAuxContainer);

            add(regSrc,  ptr[regParams + GET_OFF(srcBatchStepB)]);
            add(regGrid, ptr[regParams + GET_OFF(gridBatchStepB)]);
            add(regDst,  ptr[regParams + GET_OFF(dstBatchStepB)]);

            dec(regBatch);
            jmp(lBatchLoop, T_NEAR);
        }
        L(lEnd);
    } else {
        for (uint64_t i = 0lu; i < jcp.batchNum; i++) {
            mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);
            spatialLoop(vAuxContainer);

            add(regSrc,  jcp.srcBatchStepB);
            add(regGrid, ptr[regParams + GET_OFF(gridBatchStepB)]);
            add(regDst,  ptr[regParams + GET_OFF(dstBatchStepB)]);
        }
    }
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::spatialLoop(const Vmm* vAuxPool) {
    auto& vHCoord = vAuxPool[0];
    auto& vWCoord = vAuxPool[1];
    auto& vAux0   = vAuxPool[2];

    Xbyak::Label lSpacialLoop, lTail;
    L(lSpacialLoop);
    {
        cmp(regWorkAmount, dataElPerVec);
        jl(lTail, T_NEAR);

        getCoordinates(vHCoord, vWCoord, vAux0);
        denormalizeRawCoordinates(vWCoord, vHCoord, vAux0);
        interpolation(&vAuxPool[2], vWCoord, vHCoord);

        sub(regWorkAmount, dataElPerVec);
        add(regDst, vlen);

        jmp(lSpacialLoop, T_NEAR);
    }

    L(lTail);
    tail(vAuxPool);
}

template <>
void jitGridSampleKernel<x64::avx512_core>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord, const Vmm& vAux0) {
    uni_vpermd(vWCoord, vPermGridMask, ptr[regGrid]); // Permute to XXXX.XXXX.YYYY.YYYY
    Xbyak::Ymm ymmH = Xbyak::Ymm(vHCoord.getIdx());
    vextractf64x4(ymmH, vWCoord, 1); // Extract Y component

    add(regGrid, vlen);

    uni_vpermd(vAux0, vPermGridMask, ptr[regGrid]); // Permute to XXXX.XXXX.YYYY.YYYY
    Xbyak::Ymm ymmAux0 = Xbyak::Ymm(vAux0.getIdx());
    vinsertf64x4(vWCoord, vWCoord, ymmAux0, 1); // Extract X component
    vextractf64x4(ymmAux0, vAux0, 1); // Extract Y component
    vinsertf64x4(vHCoord, vHCoord, ymmAux0, 1);

    add(regGrid, vlen);
}

template <>
void jitGridSampleKernel<x64::sse41>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord, const Vmm& vAux0) {
    uni_vpshufd(vWCoord, ptr[regGrid], 0xD8);
    shufpd(vHCoord, vWCoord, 0x2);

    add(regGrid, vlen);

    uni_vpshufd(vAux0, ptr[regGrid], 0xD8);
    shufpd(vWCoord, vAux0, 0x0);
    shufpd(vHCoord, vAux0, 0x3);

    add(regGrid, vlen);
}

template <x64::cpu_isa_t isa> // Works for AVX, AVX2
void jitGridSampleKernel<isa>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord, const Vmm& vAux0) {
    uni_vpermd(vWCoord, vPermGridMask, ptr[regGrid]); // Permute to XXXX.YYYY
    Xbyak::Xmm xmmH = Xbyak::Xmm(vHCoord.getIdx());
    if (isa == x64::avx) { // Extract Y component
        vextractf128(xmmH, vWCoord, 1);
    } else {
        vextracti128(xmmH, vWCoord, 1);
    }

    add(regGrid, vlen);

    uni_vpermd(vAux0, vPermGridMask, ptr[regGrid]); // Permute to XXXX.YYYY
    Xbyak::Xmm xmmAux0 = Xbyak::Xmm(vAux0.getIdx());
    if (isa == x64::avx) {
        vinsertf128(vWCoord, vWCoord, xmmAux0, 1); // Extract X component
        vextractf128(xmmAux0, vAux0, 1); // Extract Y component
        vinsertf128(vHCoord, vHCoord, xmmAux0, 1);
    } else {
        vinserti128(vWCoord, vWCoord, xmmAux0, 1); // Extract X component
        vextracti128(xmmAux0, vAux0, 1); // Extract Y component
        vinserti128(vHCoord, vHCoord, xmmAux0, 1);
    }

    add(regGrid, vlen);
}

template <>
void jitGridSampleKernel<x64::avx512_core>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord, const Vmm* vAuxPool) {
    Xbyak::Label lEnd, lRest;

    auto& vAux0 = vAuxPool[0];
    Xbyak::Ymm ymmH = Xbyak::Ymm(vHCoord.getIdx());

    mov(regAux3, regWorkAmount);
    sal(regAux3, 0x1); // multiply by gridShape[3]
    cmp(regWorkAmount, dataElPerVec / 2);
    jl(lRest, T_NEAR);
    {
        uni_vpermd(vWCoord, vPermGridMask, ptr[regGrid]);
        vextractf64x4(ymmH, vWCoord, 1); // Extract Y component

        add(regGrid, vlen);
        sub(regAux3, dataElPerVec);
        cmp(regAux3, 0);
        jle(lEnd, T_NEAR);

        fillRestWorkMask(kTailMask, Xbyak::Zmm(vAux0.getIdx()), regAux3, regAux1, regAux2);
        uni_vmovups(vAux0 | kTailMask, ptr[regGrid]);
        uni_vpermd(vAux0, vPermGridMask, vAux0);
        Xbyak::Ymm ymmAux0 = Xbyak::Ymm(vAux0.getIdx());
        vinsertf64x4(vWCoord, vWCoord, ymmAux0, 1); // Extract X component
        vextractf64x4(ymmAux0, vAux0, 1); // Extract Y component
        vinsertf64x4(vHCoord, vHCoord, ymmAux0, 1);

        if (dataTypeSize > 1)
            sal(regAux3, dataTypeShift); // multiply by source data type size.
        add(regGrid, regAux3);
        jmp(lEnd, T_NEAR);
    }
    L(lRest);
    {
        fillRestWorkMask(kTailMask, Xbyak::Zmm(vAux0.getIdx()), regAux3, regAux1, regAux2);
        uni_vmovups(vWCoord | kTailMask, ptr[regGrid]);
        uni_vpermd(vWCoord, vPermGridMask, vWCoord);
        vextractf64x4(ymmH, vWCoord, 1); // Extract Y component

        if (dataTypeSize > 1)
            sal(regAux3, dataTypeShift); // multiply by source data type size.
        add(regGrid, regAux3);
    }
    L(lEnd);

    fillRestWorkMask(kTailMask, Xbyak::Zmm(vAux0.getIdx()), regWorkAmount, regAux1, regAux2);
}

template <>
void jitGridSampleKernel<x64::sse41>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord, const Vmm* vAuxPool) {
    Xbyak::Label lRest, lEnd;

    const auto& vAux = vAuxPool[0];
    const auto& vAux1 = vAuxPool[1];
    Xbyak::Xmm xmmH = Xbyak::Xmm(vHCoord.getIdx());

    mov(regAux3, regWorkAmount);
    sal(regAux3, 0x1); // multiply by gridShape[3]
    cmp(regWorkAmount, dataElPerVec / 2);
    jl(lRest, T_NEAR);
    {
        uni_vpshufd(vWCoord, ptr[regGrid], 0xD8);
        shufpd(vHCoord, vWCoord, 0x2);

        add(regGrid, vlen);
        sub(regAux3, dataElPerVec);
        cmp(regAux3, 0);
        jle(lEnd, T_NEAR);

        partialLoad32(vAux, regGrid, vAux, regAux3, regAux2);
        uni_vpshufd(vAux, vAux, 0xD8);
        shufpd(vWCoord, vAux, 0x0); // Extract X component
        shufpd(vHCoord, vAux, 0x3); // Extract Y component

        if (dataTypeSize > 1)
            sal(regAux3, dataTypeShift); // multiply by source data type size.
        add(regGrid, regAux3);
        jmp(lEnd, T_NEAR);
    }
    L(lRest);
    {
        partialLoad32(vWCoord, regGrid, vAux1, regAux3, regAux2);
        uni_vpshufd(vWCoord, vWCoord, 0xD8);  // Extract X component
        shufpd(vHCoord, vWCoord, 0x2); // Extract Y component

        if (dataTypeSize > 1)
            sal(regAux3, dataTypeShift); // multiply by source data type size.
        add(regGrid, regAux3);
    }

    L(lEnd);
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord, const Vmm* vAuxPool) {
    Xbyak::Label lRest, lEnd;

    const auto& vAux = vAuxPool[0];
    const auto& vAux1 = vAuxPool[1];
    Xbyak::Xmm xmmH = Xbyak::Xmm(vHCoord.getIdx());

    mov(regAux3, regWorkAmount);
    sal(regAux3, 0x1); // multiply by gridShape[3]
    cmp(regWorkAmount, dataElPerVec / 2);
    jl(lRest, T_NEAR);
    {
        vpermilps(vWCoord, ptr[regGrid], 0xD8);
        vextractf128(xmmH, vWCoord, 1); // Extract Y component

        add(regGrid, vlen);
        sub(regAux3, dataElPerVec);
        cmp(regAux3, 0);
        jle(lEnd, T_NEAR);

        partialLoad32(vAux, regGrid, vAux1, regAux3, regAux2);
        vpermilps(vAux, vAux, 0xD8);
        Xbyak::Xmm xmmAux = Xbyak::Xmm(vAux.getIdx());
        vinsertf128(vWCoord, vWCoord, xmmAux, 1); // Extract X component
        vextractf128(xmmAux, vAux, 1); // Extract Y component
        vinsertf128(vHCoord, vHCoord, xmmAux, 1);

        if (dataTypeSize > 1)
            sal(regAux3, dataTypeShift); // multiply by source data type size.
        add(regGrid, regAux3);
        jmp(lEnd, T_NEAR);
    }
    L(lRest);
    {
        partialLoad32(vWCoord, regGrid, vAux1, regAux3, regAux2);
        vpermilps(vWCoord, vWCoord, 0xD8);
        vextractf128(xmmH, vWCoord, 1); // Extract Y component

        if (dataTypeSize > 1)
            sal(regAux3, dataTypeShift); // multiply by source data type size.
        add(regGrid, regAux3);
    }

    L(lEnd);
}

template <>
void jitGridSampleKernel<x64::sse41>::denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord, const Vmm& vAux) {
    if (jcp.alignCorners) {
        mov(regAux4, ptr[regParams + GET_OFF(wDenormCoefF)]);
        uni_vmovups(vAux, ptr[regAux4]);
        uni_vfmadd132ps(vWCoord, vAux, vAux);

        mov(regAux4, ptr[regParams + GET_OFF(hDenormCoefF)]);
        uni_vmovups(vAux, ptr[regAux4]);
        uni_vfmadd132ps(vHCoord, vAux, vAux);
    } else {
        mov(regAux4, reinterpret_cast<uintptr_t>(halfValuesF));
        uni_vmovups(vAux, ptr[regAux4]);

        uni_vfmadd132ps(vWCoord, vSrcWidthF, vSrcWidthF);
        uni_vfmsub132ps(vWCoord, vAux, vAux);

        uni_vfmadd132ps(vHCoord, vSrcHeightF, vSrcHeightF);
        uni_vfmsub132ps(vHCoord, vAux, vAux);
    }
}

template <x64::cpu_isa_t isa> // Works for AVX, AVX2, AVX512
void jitGridSampleKernel<isa>::denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord, const Vmm& vAux) {
    if (jcp.alignCorners) {
        uni_vfmadd132ps(vWCoord, vWDenormCoefF, vWDenormCoefF);
        uni_vfmadd132ps(vHCoord, vHDenormCoefF, vHDenormCoefF);
    } else {
        uni_vfmadd132ps(vWCoord, vSrcWidthF, vSrcWidthF);
        uni_vfmsub132ps(vWCoord, vHalfF, vHalfF);

        uni_vfmadd132ps(vHCoord, vSrcHeightF, vSrcHeightF);
        uni_vfmsub132ps(vHCoord, vHalfF, vHalfF);
    }
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::interpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    if (jcp.interpolationMode == InterpolationMode::BILINEAR) {
        bilinearInterpolation(vAuxPool, vWCoord, vHCoord, tail);
    } else if (jcp.interpolationMode == InterpolationMode::BICUBIC) {
        bicubicInterpolation(vAuxPool, vWCoord, vHCoord, tail);
    } else if (jcp.interpolationMode == InterpolationMode::NEAREST) {
        nearestInterpolation(vAuxPool, vWCoord, vHCoord, tail);
    }
}

template <>
void jitGridSampleKernel<x64::avx512_core>::zerosPadding0(const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kDst, const Vmask& kAux) {
    vcmpps(kAux, vCoord, vUpperBound, 0x1);   // vCoord < vUpperBound
    vcmpps(kDst | kAux, vZeros, vCoord, 0x2); // vCoord >= vZeros
}

template <>
void jitGridSampleKernel<x64::avx512_core>::zerosPadding1(const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kDst, const Vmask& kAux) {
    vcmpps(kDst | kAux, vCoord, vUpperBound, 0x1); // vCoord < vUpperBound
    vcmpps(kDst | kDst, vZeros, vCoord, 0x2);      // vCoord >= vZeros
}

template <>
void jitGridSampleKernel<x64::avx512_core>::zerosPadding(const Vmm& vWCoord, const Vmm& vHCoord, const Vmask& kDst, const Vmask& kAux) {
    zerosPadding0(vWCoord, vSrcWidthF, kDst, kDst);
    zerosPadding1(vHCoord, vSrcHeightF, kDst, kDst);
}

template <>
void jitGridSampleKernel<x64::sse41>::zerosPadding(const Vmm& vWCoord, const Vmm& vHCoord, const Vmask& kDst, const Vmask& kAux) {
    uni_vmovups(kAux, vWCoord);
    uni_vcmpps(kAux, kAux, vSrcWidthF, 0x1); // vWCoord < vSrcWidthF
    uni_vpxor(kDst, kDst, kDst);
    uni_vcmpps(kDst, kDst, vWCoord, 0x2);    // vWCoord >= vZeros
    uni_vpand(kDst, kDst, kAux);

    uni_vmovups(kAux, vHCoord);
    uni_vcmpps(kAux, kAux, vSrcHeightF, 0x1); // vHCoord < vSrcHeightF
    uni_vpand(kDst, kDst, kAux);
    uni_vpxor(kAux, kAux, kAux);
    uni_vcmpps(kAux, kAux, vHCoord, 0x2);     // vHCoord >= vZeros
    uni_vpand(kDst, kDst, kAux);
}

template <x64::cpu_isa_t isa> // Works for AVX, AVX2
void jitGridSampleKernel<isa>::zerosPadding(const Vmm& vWCoord, const Vmm& vHCoord, const Vmask& kDst, const Vmask& kAux) {
    uni_vcmpps(kAux, vWCoord, vSrcWidthF, 0x1);  // vWCoord < vSrcWidthF
    uni_vcmpps(kDst, vZeros, vWCoord, 0x2);      // vWCoord >= vZeros
    if (isa == x64::avx)
        uni_vandps(kDst, kDst, kAux);
    else if (isa == x64::avx2)
        uni_vpand(kDst, kDst, kAux); // TODO: use uni_vandps as well?

    uni_vcmpps(kAux, vHCoord, vSrcHeightF, 0x1); // vHCoord < vSrcHeightF
    if (isa == x64::avx)
        uni_vandps(kDst, kDst, kAux);
    else if (isa == x64::avx2)
        uni_vpand(kDst, kDst, kAux);
    uni_vcmpps(kAux, vZeros, vHCoord, 0x2);      // vHCoord >= vZeros
    if (isa == x64::avx)
        uni_vandps(kDst, kDst, kAux);
    else if (isa == x64::avx2)
        uni_vpand(kDst, kDst, kAux);
}

template <>
void jitGridSampleKernel<x64::avx512_core>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmask& kAux, const coord dim) {
    Vmm vUpperBound;
    if (dim == coord::w) {
        vUpperBound = vSrcWidthSub1F;
    } else if (dim == coord::h) {
        vUpperBound = vSrcHeightSub1F;
    }
    vrangeps(vCoordDst, vCoordOrigin, vUpperBound, 0x0); // vWCoord >= vSrcWidthF
    vrangeps(vCoordDst, vCoordDst, vZeros, 0x1);         // vWCoord < vZeros
}

template <>
void jitGridSampleKernel<x64::sse41>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmask& kAux, const coord dim) {
    if (dim == coord::w) {
        mov(regAux4, ptr[regParams + GET_OFF(srcWidthSub1F)]);
    } else if (dim == coord::h) {
        mov(regAux4, ptr[regParams + GET_OFF(srcHeightSub1F)]);
    }
    uni_vmovups(kAux, vCoordOrigin);
    uni_vcmpps(kAux, kAux, ptr[regAux4], 0x2); // vCoord < vUpperBound
    if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
        uni_vmovups(vCoordDst, vCoordOrigin);
    uni_vpand(vCoordDst, vCoordDst, kAux);
    uni_vpand(kAux, kAux, ptr[regAux4]);
    uni_vaddps(vCoordDst, vCoordDst, kAux);

    uni_vpxor(kAux, kAux, kAux);
    uni_vcmpps(kAux, kAux, vCoordDst, 0x2);    // vCoord >= vZeros
    uni_vpand(vCoordDst, vCoordDst, kAux);
}

template <x64::cpu_isa_t isa> // Works for AVX, AVX2
void jitGridSampleKernel<isa>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmask& kAux, const coord dim) {
    if (dim == coord::w) {
        mov(regAux4, ptr[regParams + GET_OFF(srcWidthSub1F)]);
    } else if (dim == coord::h) {
        mov(regAux4, ptr[regParams + GET_OFF(srcHeightSub1F)]);
    }
    uni_vcmpps(kAux, vCoordOrigin, ptr[regAux4], 0x2); // vCoord < vUpperBound
    uni_vpand(vCoordDst, vCoordOrigin, kAux);
    uni_vpand(kAux, kAux, ptr[regAux4]);
    uni_vaddps(vCoordDst, vCoordDst, kAux);

    uni_vcmpps(kAux, vZeros, vCoordDst, 0x2); // vCoord >= vZeros
    uni_vpand(vCoordDst, vCoordDst, kAux);
}

template <>
void jitGridSampleKernel<x64::avx512_core>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmm& vAux, const Vmask& kAux, const coord dim) {
    Vmm vSrcDimF, vSrcDimMul2F, vSrcDimMul2Sub1F;
    if (dim == coord::w) {
        vSrcDimF = vSrcWidthF;
        vSrcDimMul2F = vSrcWidthMul2F;
        vSrcDimMul2Sub1F = vSrcWidthMul2Sub1F;
    } else if (coord::h) {
        vSrcDimF = vSrcHeightF;
        vSrcDimMul2F = vSrcHeightMul2F;
        vSrcDimMul2Sub1F = vSrcHeightMul2Sub1F;
    }

    if (jcp.alignCorners) {
        // abs(x) % D21
        uni_vandps(vCoordDst, vCoordOrigin, vAbsMask); // abs(x)
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2Sub1F);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2Sub1F); // abs(x) % D21
    } else {
        // (x % D2 + D2) % D2
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2F);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2F); // x % D2
        uni_vaddps(vCoordDst, vCoordDst, vSrcDimMul2F); // x % D2 + D2
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2F);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2F); // (x % D2 + D2) % D2
    }

    uni_vsubps(vAux, vSrcDimMul2Sub1F, vCoordDst);
    vcmpps(kAux, vSrcDimF, vCoordDst, 0x2); // vCoordDst >= vSrcDimF
    vmovups(vCoordDst | kAux, vAux);
}

template <>
void jitGridSampleKernel<x64::sse41>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmm& vAux, const Vmask& kAux, const coord dim) {
    if (jcp.alignCorners) {
        // abs(x) % D21
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        mov(regAux4, reinterpret_cast<uintptr_t>(absMask));
        uni_vandps(vCoordDst, vCoordDst, ptr[regAux4]); // abs(x)
        if (dim == coord::w) {
            mov(regAux4, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
        } else if (coord::h) {
            mov(regAux4, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
        }
        uni_vmovups(vAux, vCoordDst);
        uni_vdivps(vAux, vAux, ptr[regAux4]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[regAux4]); // abs(x) % D21
    } else {
        // (x % D2 + D2) % D2
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        uni_vmovups(vAux, vCoordOrigin);
        if (dim == coord::w) {
            mov(regAux4, ptr[regParams + GET_OFF(srcWidthMul2F)]);
        } else if (coord::h) {
            mov(regAux4, ptr[regParams + GET_OFF(srcHeightMul2F)]);
        }
        uni_vdivps(vAux, vAux, ptr[regAux4]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[regAux4]); // x % D2
        uni_vaddps(vCoordDst, vCoordDst, ptr[regAux4]); // x % D2 + D2
        uni_vmovups(vAux, vCoordDst);
        uni_vdivps(vAux, vAux, ptr[regAux4]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[regAux4]); // (x % D2 + D2) % D2
    }

    if (dim == coord::w) {
        mov(regAux4, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
    } else if (coord::h) {
        mov(regAux4, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
    }
    uni_vmovups(vAux, ptr[regAux4]);
    uni_vsubps(vAux, vAux, vCoordDst);
    if (dim == coord::w) {
        uni_vmovups(kAux, vSrcWidthF);
    } else if (coord::h) {
        uni_vmovups(kAux, vSrcHeightF);
    }
    vcmpps(kAux, kAux, vCoordDst, 0x2); // vCoordDst >= vSrcDimF
    uni_vpand(vCoordDst, vCoordDst, kAux);
    uni_vpand(kAux, kAux, vAux);
    uni_vaddps(vCoordDst, vCoordDst, kAux);
}

template <x64::cpu_isa_t isa> // Works for AVX, AVX2
void jitGridSampleKernel<isa>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmm& vAux, const Vmask& kAux, const coord dim) {
    if (jcp.alignCorners) {
        // abs(x) % D21
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        mov(regAux4, reinterpret_cast<uintptr_t>(absMask));
        uni_vandps(vCoordDst, vCoordDst, ptr[regAux4]); // abs(x)
        if (dim == coord::w) {
            mov(regAux4, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
        } else if (coord::h) {
            mov(regAux4, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
        }
        uni_vmovups(vAux, vCoordDst);
        uni_vdivps(vAux, vAux, ptr[regAux4]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[regAux4]); // abs(x) % D21
    } else {
        // (x % D2 + D2) % D2
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        uni_vmovups(vAux, vCoordOrigin);
        if (dim == coord::w) {
            mov(regAux4, ptr[regParams + GET_OFF(srcWidthMul2F)]);
        } else if (coord::h) {
            mov(regAux4, ptr[regParams + GET_OFF(srcHeightMul2F)]);
        }
        uni_vdivps(vAux, vAux, ptr[regAux4]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[regAux4]); // x % D2
        uni_vaddps(vCoordDst, vCoordDst, ptr[regAux4]); // x % D2 + D2
        uni_vmovups(vAux, vCoordDst);
        uni_vdivps(vAux, vAux, ptr[regAux4]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[regAux4]); // (x % D2 + D2) % D2
    }

    if (dim == coord::w) {
        mov(regAux4, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
    } else if (coord::h) {
        mov(regAux4, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
    }
    uni_vmovups(vAux, ptr[regAux4]);
    uni_vsubps(vAux, vAux, vCoordDst);
    if (dim == coord::w) {
        uni_vmovups(kAux, vSrcWidthF);
    } else if (coord::h) {
        uni_vmovups(kAux, vSrcHeightF);
    }
    vcmpps(kAux, kAux, vCoordDst, 0x2); // vCoordDst >= vSrcDimF
    uni_vpand(vCoordDst, vCoordDst, kAux);
    uni_vpand(kAux, kAux, vAux);
    uni_vaddps(vCoordDst, vCoordDst, kAux);
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::bicubicCoefficients(const Vmm& vCoef, const Vmm& vDDim, const uint8_t idx) {
    if (idx == 0) {
        uni_vmovups(vCoef, vDDim);
        vfnmadd132ps(vCoef, vOnesF, v2val);
        uni_vfmadd231ps(vCoef, vDDim, vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vmulps(vCoef, vCoef, vBicubConst);
    } else if (idx == 1) {
        uni_vmovups(vCoef, vDDim);
        vfmsub132ps(vCoef, vBicub3Const, vBicub2Const);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vfmadd132ps(vCoef, vOnesF, vDDim);
    } else if (idx == 2) {
        uni_vmulps(vCoef, vDDim, vDDim);
        vfmadd132ps(vCoef, vBicubConst, vBicub2Const);
        vfmsub231ps(vCoef, vDDim, v2Bicub3Const);
        uni_vmulps(vCoef, vCoef, vDDim);
    } else if (idx == 3) {
        uni_vmulps(vCoef, vBicubConst, vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
        vfnmadd132ps(vCoef, vCoef, vDDim);
    }
}

template <>
void jitGridSampleKernel<x64::avx512_core>::nearestInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    const auto& vSrcShift  = vWCoord;
    const auto& vAux       = vAuxPool[0];
    const auto& kMask0   = k1;
    const auto& kAuxMask = k2;

    uni_vroundps(vWCoord, vWCoord, 0x0); // Round near
    uni_vroundps(vHCoord, vHCoord, 0x0); // Round near

    if (jcp.paddingMode == PaddingMode::ZEROS) {
        zerosPadding(vWCoord, vHCoord, kMask0, kAuxMask);
    } else if (jcp.paddingMode == PaddingMode::BORDER) {
        borderPadding(vWCoord, vWCoord, kAuxMask, coord::w);
        borderPadding(vHCoord, vHCoord, kAuxMask, coord::h);
    } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
        reflectionPadding(vWCoord, vWCoord, vAux, kAuxMask, coord::w);
        reflectionPadding(vHCoord, vHCoord, vAux, kAuxMask, coord::h);
    }

    hwShiftPs2dq(vSrcShift, vHCoord, vWCoord, vSrcWidthF);

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    const Xbyak::Reg64& rChannel = regAux1;
    const Xbyak::Reg64& rSrcTmp  = regAux2;
    const Xbyak::Reg64& rDstTmp  = regAux3;
    mov(rChannel, 0);
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, regChannelsNum);
        jge(lChannelLoopEnd, T_NEAR);

        if (jcp.paddingMode == PaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask0);
            uni_vpxor(vAux, vAux, vAux);
        } else {
            kxnorw(kAuxMask, kAuxMask, kAuxMask);
        }

        uni_vpgatherdd(vAux, ptr[rSrcTmp + vSrcShift], kAuxMask);

        if (tail) {
            uni_vmovups(ptr[rDstTmp] | kTailMask, vAux);
        } else {
            uni_vmovups(ptr[rDstTmp], vAux);
        }
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        inc(rChannel);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <x64::cpu_isa_t isa> // Works for AVX2, AVX, SSE41
void jitGridSampleKernel<isa>::nearestInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    const auto& vSrcShift = vWCoord;
    const auto& vAux      = vAuxPool[0];
    const auto& kMask     = vAuxPool[1];

    uni_vroundps(vWCoord, vWCoord, 0x0); // Round near
    uni_vroundps(vHCoord, vHCoord, 0x0); // Round near

    bool useMask = tail, zeroMask = false;
    if (jcp.paddingMode == PaddingMode::ZEROS) {
        zerosPadding(vWCoord, vHCoord, kMask, vAux);
        useMask = true;
        zeroMask = true;
    } else if (jcp.paddingMode == PaddingMode::BORDER) {
        borderPadding(vWCoord, vWCoord, kMask, coord::w);
        borderPadding(vHCoord, vHCoord, kMask, coord::h);
    } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
        reflectionPadding(vWCoord, vWCoord, vAux, kMask, coord::w);
        reflectionPadding(vHCoord, vHCoord, vAux, kMask, coord::h);
    }

    hwShiftPs2dq(vSrcShift, vHCoord, vWCoord, vSrcWidthF);

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    const Xbyak::Reg64& rChannel = regAux1;
    const Xbyak::Reg64& rSrcTmp  = regAux2;
    const Xbyak::Reg64& rDstTmp  = regAux3;
    mov(rChannel, ptr[regParams + GET_OFF(channelsNum)]);
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, 0);
        jle(lChannelLoopEnd, T_NEAR);

        maskMov32(rDstTmp, rSrcTmp, kMask, kMask, vSrcShift, vAux, regAux4, useMask, zeroMask);

        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        dec(rChannel);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <>
void jitGridSampleKernel<x64::avx512_core>::bilinearInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    const auto &vDX     = vAuxPool[0];
    const auto &vDY     = vAuxPool[1];
    const auto &vQ0     = vAuxPool[2];
    const auto &vQ1     = vAuxPool[3];
    const auto &shift00 = vWCoord;
    const auto &shift01 = vHCoord;
    const auto &shift10 = vAuxPool[4];
    const auto &shift11 = vAuxPool[5];
    const auto &vAux3   = vAuxPool[6];
    const auto &kMask00  = k1;
    const auto &kMask01  = k2;
    const auto &kMask10  = k3;
    const auto &kMask11  = k4;
    const auto &kAuxMask = k5;

    uni_vmovups(vDX, vWCoord);
    uni_vmovups(vDY, vHCoord);
    uni_vroundps(vWCoord, vWCoord, 0x1); // Round floor
    uni_vroundps(vHCoord, vHCoord, 0x1); // Round floor
    uni_vsubps(vDX, vDX, vWCoord);
    uni_vsubps(vDY, vDY, vHCoord);

    uni_vaddps(shift10, vWCoord, vOnesF);
    uni_vaddps(shift11, vHCoord, vOnesF);
    if (jcp.paddingMode == PaddingMode::ZEROS) {
        zerosPadding(vWCoord, vHCoord, kMask00, kAuxMask); // (y; x)
        zerosPadding(shift10, vHCoord, kMask01, kAuxMask); // (y; x + 1)
        zerosPadding(shift10, shift11, kMask11, kAuxMask); // (y + 1; x + 1)
        zerosPadding(vWCoord, shift11, kMask10, kAuxMask); // (y + 1; x)

        hwShiftPs2dq(shift00, vHCoord, vWCoord, vSrcWidthF);
        uni_vpaddd(shift01, shift00, vDataTypeSize);
        uni_vpaddd(shift10, shift00, vSrcWidthB); // shift11??
        uni_vpaddd(shift11, shift10, vDataTypeSize); // sub??
    } else if (jcp.paddingMode == PaddingMode::BORDER) {
        borderPadding(vWCoord, vWCoord, kAuxMask, coord::w);
        borderPadding(vHCoord, vHCoord, kAuxMask, coord::h);
        borderPadding(shift10, shift10, kAuxMask, coord::w);
        borderPadding(shift11, shift11, kAuxMask, coord::h);
    } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
        reflectionPadding(vWCoord, vWCoord, vQ0, kAuxMask, coord::w);
        reflectionPadding(vHCoord, vHCoord, vQ0, kAuxMask, coord::h);
        reflectionPadding(shift10, shift10, vQ0, kAuxMask, coord::w);
        reflectionPadding(shift11, shift11, vQ0, kAuxMask, coord::h);
    }
    if (jcp.paddingMode == PaddingMode::BORDER || jcp.paddingMode == PaddingMode::REFLECTION) {
        uni_vmovups(vAux3, shift11);
        // W * y + x
        uni_vfmadd132ps(vAux3, vWCoord, vSrcWidthF);   // (y + 1; x)
        uni_vfmadd231ps(vWCoord, vHCoord, vSrcWidthF); // (y; x)
        uni_vfmadd132ps(vHCoord, shift10, vSrcWidthF); // (y; x + 1)
        uni_vfmadd132ps(shift11, shift10, vSrcWidthF); // (y + 1; x + 1)
        uni_vcvtps2dq(shift00, vWCoord);
        uni_vcvtps2dq(shift01, vHCoord);
        uni_vcvtps2dq(shift10, vAux3);
        uni_vcvtps2dq(shift11, shift11);
        if (dataTypeSize > 1) {
            uni_vpslld(shift00, shift00, dataTypeShift);
            uni_vpslld(shift01, shift01, dataTypeShift);
            uni_vpslld(shift10, shift10, dataTypeShift);
            uni_vpslld(shift11, shift11, dataTypeShift);
        }
    }

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    const Xbyak::Reg64 &rChannel = regAux1;
    const Xbyak::Reg64 &rSrcTmp  = regAux2;
    const Xbyak::Reg64 &rDstTmp  = regAux3;
    mov(rChannel, 0);
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, regChannelsNum);
        jge(lChannelLoopEnd, T_NEAR);

        // (y; x)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask00);
            uni_vpxor(vQ0, vQ0, vQ0);
        } else {
            kxnorw(kAuxMask, kAuxMask, kAuxMask);
        }
        uni_vpgatherdd(vQ0, ptr[rSrcTmp + shift00], kAuxMask); // v00 -> vQ0
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vQ0, vQ0);
        }
        uni_vfmsub213ps(vQ0, vDX, vQ0); // q0 = -(v00 - dx * v00)

        // (y; x + 1)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask01);
            uni_vpxor(vAux3, vAux3, vAux3);
        } else {
            kxnorw(kAuxMask, kAuxMask, kAuxMask);
        }
        uni_vpgatherdd(vAux3, ptr[rSrcTmp + shift01], kAuxMask);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vAux3, vAux3);
        }
        uni_vfmsub231ps(vQ0, vAux3, vDX); // q0 = -q0 + dx * v01

        // (y + 1; x + 1)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask11);
            uni_vpxor(vAux3, vAux3, vAux3);
        } else {
            kxnorw(kAuxMask, kAuxMask, kAuxMask);
        }
        uni_vpgatherdd(vAux3, ptr[rSrcTmp + shift11], kAuxMask);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vAux3, vAux3);
        }

        // (y + 1; x)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask10);
            uni_vpxor(vQ1, vQ1, vQ1);
        } else {
            kxnorw(kAuxMask, kAuxMask, kAuxMask);
        }
        uni_vpgatherdd(vQ1, ptr[rSrcTmp + shift10], kAuxMask);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vQ1, vQ1);
        }

        uni_vfmsub213ps(vQ1, vDX, vQ1); // q1 = -(v10 - dx * v10)
        uni_vfmsub231ps(vQ1, vAux3, vDX); // q1 = -q1 + dx * v11
        // Res = q0 + dy * (q1 - q0)
        uni_vsubps(vQ1, vQ1, vQ0);
        uni_vfmadd132ps(vQ1, vQ0, vDY);

        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtps2dq(vQ1, vQ1);
        }

        if (tail) {
            uni_vmovups(ptr[rDstTmp] | kTailMask, vQ1);
        } else {
            uni_vmovups(ptr[rDstTmp], vQ1);
        }
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        add(rChannel, 1);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::bilinearInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    const auto& vDX = vAuxPool[0];
    const auto& vDY = vAuxPool[1];
    const auto& vGatherShift = vAuxPool[2];
    const auto& vAux3 = vAuxPool[3];
    const auto& vQ0 = vAuxPool[4];
    const auto& vQ1 = vAuxPool[5];
    const auto& kMask0 = masksContainer[4];
    const auto& kMask1 = masksContainer[5];

    uni_vmovups(vDX, vWCoord);
    uni_vmovups(vDY, vHCoord);
    uni_vroundps(vWCoord, vWCoord, 0x1); // Round floor
    uni_vroundps(vHCoord, vHCoord, 0x1); // Round floor
    uni_vsubps(vDX, vDX, vWCoord);
    uni_vsubps(vDY, vDY, vHCoord);

    if (jcp.paddingMode == PaddingMode::ZEROS) {
        zerosPadding(vWCoord, vHCoord, kMask0, kMask1);
    }

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    const Xbyak::Reg64& rChannel = regAux1;
    const Xbyak::Reg64& rSrcTmp = regAux2;
    const Xbyak::Reg64& rDstTmp = regAux3;
    mov(rChannel, 0);
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, regChannelsNum);
        jge(lChannelLoopEnd, T_NEAR);

        if (jcp.paddingMode == PaddingMode::ZEROS) {
            // (x; y)
            zerosPadding(vWCoord, vHCoord, kMask0, kMask1);
            uni_vmovups(vGatherShift, vHCoord);
            hwShiftPs2dq(vGatherShift, vGatherShift, vWCoord, vSrcWidthF);
            uni_vpxor(vQ0, vQ0, vQ0);
            uni_vpgatherdd(vQ0, ptr[rSrcTmp + vGatherShift], kMask0); // v00 -> vQ0
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vQ0, vQ0);
            }
            uni_vfmsub213ps(vQ0, vDX, vQ0); // q0 = -(v00 - dx * v00)

            // (x + 1; y)
            uni_vaddps(vWCoord, vWCoord, vOnesF);
            zerosPadding(vWCoord, vHCoord, kMask0, kMask1);
            uni_vmovups(vGatherShift, vHCoord);
            hwShiftPs2dq(vGatherShift, vGatherShift, vWCoord, vSrcWidthF);
            uni_vpxor(vAux3, vAux3, vAux3);
            uni_vpgatherdd(vAux3, ptr[rSrcTmp + vGatherShift], kMask0);

            // q0 = -q0 + dx * v01
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux3, vAux3);
            }
            uni_vfmsub231ps(vQ0, vAux3, vDX);

            // (x + 1; y + 1)
            uni_vaddps(vHCoord, vHCoord, vOnesF);
            zerosPadding(vWCoord, vHCoord, kMask0, kMask1);
            uni_vmovups(vGatherShift, vHCoord);
            hwShiftPs2dq(vGatherShift, vGatherShift, vWCoord, vSrcWidthF);
            uni_vpxor(vAux3, vAux3, vAux3);
            uni_vpgatherdd(vAux3, ptr[rSrcTmp + vGatherShift], kMask0);

            // (x; y + 1)
            uni_vsubps(vWCoord, vWCoord, vOnesF);
            zerosPadding(vWCoord, vHCoord, kMask0, kMask1);
            uni_vmovups(vGatherShift, vHCoord);
            hwShiftPs2dq(vGatherShift, vGatherShift, vWCoord, vSrcWidthF);
            uni_vpxor(vQ1, vQ1, vQ1);
            uni_vpgatherdd(vQ1, ptr[rSrcTmp + vGatherShift], kMask0);
            uni_vsubps(vHCoord, vHCoord, vOnesF);
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vQ1, vQ1);
            }

            uni_vfmsub213ps(vQ1, vDX, vQ1); // q1 = -(v10 - dx * v10)
            uni_vfmsub231ps(vQ1, vAux3, vDX); // q1 = -q1 + dx * v11
            // Res = q0 + dy * (q1 - q0)
            uni_vsubps(vQ1, vQ1, vQ0);
            uni_vfmadd132ps(vQ1, vQ0, vDY);

            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtps2dq(vQ1, vQ1);
            }
        }

        uni_vmovups(ptr[rDstTmp], vQ1);
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        add(rChannel, 1);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <>
void jitGridSampleKernel<x64::avx512_core>::bicubicInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    const auto& vHTop      = vAuxPool[0];
    const auto& vWLeft     = vAuxPool[1];
    const auto& vDX        = vAuxPool[2];
    const auto& vDY        = vAuxPool[3];
    const auto& vXDotProd  = vAuxPool[4];
    const auto& vYDotProd  = vDX;
    const auto& vSrcShift0 = vAuxPool[5];
    const auto& vSrcShift  = vAuxPool[6];
    const auto& vCX0       = vAuxPool[7];
    const auto& vCX1       = vAuxPool[8];
    const auto& vCX2       = vAuxPool[9];
    const auto& vCX3       = vAuxPool[10];
    const auto& vAux       = vAuxPool[11];
    const auto& kMask0   = masksContainer[1];
    const auto& kMask1   = masksContainer[2];
    const auto& kMask2   = masksContainer[3];
    const auto& kMask3   = masksContainer[4];
    const auto& kAuxMask = masksContainer[5];
    const auto& kMaskH   = masksContainer[6];

    uni_vroundps(vHTop, vHCoord, 0x1);  // Round floor
    uni_vroundps(vWLeft, vWCoord, 0x1); // Round floor
    uni_vsubps(vDY, vHCoord, vHTop);
    uni_vsubps(vDX, vWCoord, vWLeft);
    uni_vsubps(vHTop, vHTop, vOnesF);
    uni_vsubps(vWLeft, vWLeft, vOnesF);

    for (int i = 0; i < 4; i++) {
        bicubicCoefficients((&vCX0)[i], vDX, i);
    }

    if (jcp.paddingMode == PaddingMode::ZEROS) {
        zerosPadding0(vWLeft, vSrcWidthF, kMask0, kAuxMask);
        uni_vaddps(vWCoord, vWLeft, vOnesF);
        zerosPadding0(vWCoord, vSrcWidthF, kMask1, kAuxMask);
        uni_vaddps(vWCoord, vWCoord, vOnesF);
        zerosPadding0(vWCoord, vSrcWidthF, kMask2, kAuxMask);
        uni_vaddps(vWCoord, vWCoord, vOnesF);
        zerosPadding0(vWCoord, vSrcWidthF, kMask3, kAuxMask);
    }

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    const Xbyak::Reg64& rChannel = regAux1;
    const Xbyak::Reg64& rSrcTmp  = regAux2;
    const Xbyak::Reg64& rDstTmp  = regAux3;
    mov(rChannel, 0);
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, regChannelsNum);
        jge(lChannelLoopEnd, T_NEAR);

        uni_vmovups(vHCoord, vHTop);
        uni_vpxor(vYDotProd, vYDotProd, vYDotProd);
        for (int h = 0; h < 4; h++) {
            // (y - 1 + i; x - 1)
            if (jcp.paddingMode == PaddingMode::ZEROS) {
                zerosPadding0(vHCoord, vSrcHeightF, kMaskH, kMaskH);
                kandw(kAuxMask, kMaskH, kMask0);
                uni_vmulps(vSrcShift0, vHCoord, vSrcWidthF);
                uni_vmovups(vWCoord, vWLeft);
                uni_vaddps(vSrcShift, vSrcShift0, vWCoord);
                uni_vpxor(vAux, vAux, vAux);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                borderPadding(vSrcShift0, vHCoord, kAuxMask, coord::h);
                uni_vmulps(vSrcShift0, vSrcShift0, vSrcWidthF);
                uni_vmovups(vWCoord, vWLeft);
                borderPadding(vSrcShift, vWCoord, kAuxMask, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                kxnorw(kAuxMask, kAuxMask, kAuxMask);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                reflectionPadding(vSrcShift0, vHCoord, vAux, kAuxMask, coord::h);
                uni_vmulps(vSrcShift0, vSrcShift0, vSrcWidthF);
                uni_vmovups(vWCoord, vWLeft);
                reflectionPadding(vSrcShift, vWCoord, vAux, kAuxMask, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                kxnorw(kAuxMask, kAuxMask, kAuxMask);
            }
            uni_vcvtps2dq(vSrcShift, vSrcShift);
            if (dataTypeSize > 1)
                uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            uni_vpgatherdd(vAux, ptr[rSrcTmp + vSrcShift], kAuxMask);
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vmulps(vXDotProd, vAux, vCX0);

            // (y - 1 + i; x)
            // (y - 1 + i; x + 1)
            // (y - 1 + i; x + 2)
            for (int w = 1; w < 4; w++) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                if (jcp.paddingMode == PaddingMode::ZEROS) {
                    uni_vaddps(vSrcShift, vSrcShift0, vWCoord);
                    kandw(kAuxMask, kMaskH, (&kMask0)[w]);
                    uni_vpxor(vAux, vAux, vAux);
                } else if (jcp.paddingMode == PaddingMode::BORDER) {
                    borderPadding(vSrcShift, vWCoord, kAuxMask, coord::w);
                    uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                    kxnorw(kAuxMask, kAuxMask, kAuxMask);
                } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                    reflectionPadding(vSrcShift, vWCoord, vAux, kAuxMask, coord::w);
                    uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                    kxnorw(kAuxMask, kAuxMask, kAuxMask);
                }
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
                uni_vpgatherdd(vAux, ptr[rSrcTmp + vSrcShift], kAuxMask);
                if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                    uni_vcvtdq2ps(vAux, vAux);
                }
                uni_vfmadd231ps(vXDotProd, vAux, (&vCX0)[w]);
            }

            if (h != 3) {
                uni_vaddps(vHCoord, vHCoord, vOnesF);
            }

            bicubicCoefficients(vAux, vDY, h);
            uni_vfmadd231ps(vYDotProd, vXDotProd, vAux);
        }

        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtps2dq(vYDotProd, vYDotProd);
        }

        if (tail) {
            uni_vmovups(ptr[rDstTmp] | kTailMask, vYDotProd);
        } else {
            uni_vmovups(ptr[rDstTmp], vYDotProd);
        }
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        add(rChannel, 1);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::bicubicInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    const auto& vHTop      = vAuxPool[0];
    const auto& vWLeft     = vAuxPool[1];
    const auto& vDX        = vAuxPool[2];
    const auto& vDY        = vAuxPool[3];
    const auto& vXDotProd  = vAuxPool[4];
    const auto& vYDotProd  = vDX;
    const auto& vSrcShift0 = vAuxPool[5];
    const auto& vSrcShift  = vAuxPool[6];
    const auto& vCX0       = vAuxPool[7];
    const auto& vCX1       = vAuxPool[8];
    const auto& vCX2       = vAuxPool[9];
    const auto& vCX3       = vAuxPool[10];
    const auto& vAux       = vAuxPool[11]; // &vWLeft
    const auto& kMask0   = vCX0;
    const auto& kMask1   = vCX1;
    const auto& kMask2   = vCX2;
    const auto& kMask3   = vCX3;
    const auto& kAuxMask = vAux;
    const auto& kMaskH   = vSrcShift;

    uni_vroundps(vWLeft, vWCoord, 0x1); // Round floor
    uni_vroundps(vHTop, vHCoord, 0x1);  // Round floor
    uni_vsubps(vDX, vWCoord, vWLeft);
    uni_vsubps(vDY, vHCoord, vHTop);
    uni_vsubps(vWLeft, vWLeft, vOnesF);
    uni_vsubps(vHTop, vHTop, vOnesF);

    bicubicCoefficients(vCX0, vDX, 0); // TODO: for
    bicubicCoefficients(vCX1, vDX, 1);
    bicubicCoefficients(vCX2, vDX, 2);
    bicubicCoefficients(vCX3, vDX, 3);

    if (jcp.paddingMode == PaddingMode::ZEROS) {
        hwShiftPs2dq(vSrcShift0, vHTop, vWLeft, vSrcWidthF);
    }

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    const Xbyak::Reg64& rChannel = regAux1;
    const Xbyak::Reg64& rSrcTmp = regAux2;
    const Xbyak::Reg64& rDstTmp = regAux3;
    mov(rChannel, 0);
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, regChannelsNum);
        jge(lChannelLoopEnd, T_NEAR);

        if (jcp.paddingMode == PaddingMode::ZEROS) {
            uni_vmovups(vSrcShift, vSrcShift0);
        }
        uni_vmovups(vHCoord, vHTop);
        uni_vpxor(vYDotProd, vYDotProd, vYDotProd);
        for (int i = 0; i < 4; i++) {
            // (y - 1 + i; x - 1)
            if (jcp.paddingMode == PaddingMode::ZEROS) {
                zerosPadding(vHCoord, vSrcHeightF, kMaskH, kMaskH);
                uni_vpxor(vAux, vAux, vAux);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                borderPadding(vSrcShift0, vHCoord, kAuxMask, coord::h);
                uni_vmulps(vSrcShift0, vSrcShift0, vSrcWidthF);
                uni_vmovups(vWCoord, vWLeft);
                borderPadding(vSrcShift, vWCoord, kAuxMask, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                reflectionPadding(vSrcShift0, vHCoord, vAux, kAuxMask, coord::h);
                uni_vmulps(vSrcShift0, vSrcShift0, vSrcWidthF);
                uni_vmovups(vWCoord, vWLeft);
                reflectionPadding(vSrcShift, vWCoord, vAux, kAuxMask, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            }
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vmulps(vXDotProd, vAux, vCX0);
// TODO: cycle 3?
            // (y - 1 + i; x)
            if (jcp.paddingMode == PaddingMode::ZEROS) {
                uni_vpxor(vAux, vAux, vAux);
                uni_vpaddd(vSrcShift, vSrcShift, vDataTypeSize);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                borderPadding(vSrcShift, vWCoord, kAuxMask, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                reflectionPadding(vSrcShift, vWCoord, vAux, kAuxMask, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            }
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vfmadd231ps(vXDotProd, vAux, vCX1);

            // (y - 1 + i; x + 1)
            if (jcp.paddingMode == PaddingMode::ZEROS) {
                uni_vpxor(vAux, vAux, vAux);
                uni_vpaddd(vSrcShift, vSrcShift, vDataTypeSize);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                borderPadding(vSrcShift, vWCoord, kAuxMask, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                reflectionPadding(vSrcShift, vWCoord, vAux, kAuxMask, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            }
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vfmadd231ps(vXDotProd, vAux, vCX2);

            // (y - 1 + i; x + 2)
            if (jcp.paddingMode == PaddingMode::ZEROS) {
                uni_vpxor(vAux, vAux, vAux);
                uni_vpaddd(vSrcShift, vSrcShift, vDataTypeSize);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                borderPadding(vSrcShift, vWCoord, kAuxMask, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                reflectionPadding(vSrcShift, vWCoord, vAux, kAuxMask, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            }
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vfmadd231ps(vXDotProd, vAux, vCX3);

            if (i != 3) {
                uni_vaddps(vHCoord, vHCoord, vOnesF);
                if (jcp.paddingMode == PaddingMode::ZEROS) {
                    uni_vpaddd(vSrcShift, vSrcShift, vSrcWidthB);
                }
            }

            bicubicCoefficients(vAux, vDY, i);
            uni_vfmadd231ps(vYDotProd, vXDotProd, vAux);
        }

        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtps2dq(vYDotProd, vYDotProd);
        }

        uni_vmovups(ptr[rDstTmp], vYDotProd);
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        add(rChannel, 1);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::tail(const Vmm* vAuxPool) {
    Xbyak::Label lEnd;
    cmp(regWorkAmount, 0);
    jle(lEnd, T_NEAR);

    auto& vHCoord = vAuxPool[0];
    auto& vWCoord = vAuxPool[1];
    auto& vAux0   = vAuxPool[2];

    getTailCoordinates(vHCoord, vWCoord, &vAuxPool[2]);
    denormalizeRawCoordinates(vWCoord, vHCoord, vAux0);
    interpolation(&vAuxPool[2], vWCoord, vHCoord, true);

    if (dataTypeSize > 1)
        sal(regWorkAmount, dataTypeShift); // multiply by source data type size.
    add(regDst, regWorkAmount);

    L(lEnd);
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::hwShiftPs2dq(const Vmm& vDst, const Vmm& vHCoord,const Vmm& vWCoord, const Vmm& vWidth) {
    if (vDst.getIdx() == vWCoord.getIdx()) {
        uni_vfmadd231ps(vDst, vHCoord, vWidth);
    } else if (vDst.getIdx() == vHCoord.getIdx()) {
        uni_vfmadd132ps(vDst, vWCoord, vWidth);
    } else if (vDst.getIdx() == vWidth.getIdx()) {
        uni_vfmadd132ps(vDst, vWCoord, vHCoord);
    } else {
        uni_vmovups(vDst, vWCoord);
        uni_vfmadd231ps(vDst, vHCoord, vWidth);
    }

    if (isa == x64::avx) { // vpslld works just with XMM for AVX, so use vmulps for YMM
        if (dataTypeSize > 1) {
            mov(regAux1, reinterpret_cast<uintptr_t>(dataTypeSizeArr));
            uni_vmulps(vDst, vDst, ptr[regAux1]);
        }
        uni_vcvtps2dq(vDst, vDst);
    } else {
        uni_vcvtps2dq(vDst, vWCoord);
        if (dataTypeSize > 1)
            uni_vpslld(vDst, vDst, dataTypeShift); // multiply by source data type size.
    }
}

template struct jitGridSampleKernel<x64::avx512_core>;
template struct jitGridSampleKernel<x64::avx2>;
template struct jitGridSampleKernel<x64::avx>;
template struct jitGridSampleKernel<x64::sse41>;

}   // namespace intel_cpu
}   // namespace ov
