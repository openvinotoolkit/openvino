// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grid_sample.hpp"

using namespace dnnl::impl::cpu;

namespace ov {
namespace intel_cpu {
namespace kernel {

#define GET_OFF(field) offsetof(GridSamplesKernelExecArgs, field)

template <x64::cpu_isa_t isa>
GridSampleKernel<isa>::GridSampleKernel(const GridSampleKernelConfParams& jcp) :
        GridSampleKernelBase(jit_name(), jcp, isa) {
    vlen = x64::cpu_isa_traits<isa>::vlen;
    dataTypeSize = jcp.inDataPrc.size();
    gridTypeSize = jcp.gridPrc.size();
    dataElPerVec = vlen / dataTypeSize;
    gridElPerVec = vlen / gridTypeSize;
    if (dataTypeSize == 2)
        dataTypeShift = 1;
    else if (dataTypeSize == 4)
        dataTypeShift = 2;
}

template <x64::cpu_isa_t isa>
void GridSampleKernel<isa>::create_ker() {
    auto code = x64::jit_generator::create_kernel();
    if (code != dnnl::impl::status::success)
        OPENVINO_THROW("Could not create GridSample kernel. Error code: ", std::to_string(code));
    ker_ = (decltype(ker_))jit_ker();
}

template <x64::cpu_isa_t isa>
void GridSampleKernel<isa>::generate() {
    this->preamble();
    registersPool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

    regSrc  = getReg64();
    regGrid = getReg64();
    regDst  = getReg64();
    regSrcChannelStepB = getReg64();
    regDstChannelStepB = getReg64();

    mov(regSrc,  ptr[regParams + GET_OFF(src)]);
    mov(regGrid, ptr[regParams + GET_OFF(grid)]);
    mov(regDst,  ptr[regParams + GET_OFF(dst)]);
    mov(regSrcChannelStepB, ptr[regParams + GET_OFF(srcChannelStepB)]);
    mov(regDstChannelStepB, ptr[regParams + GET_OFF(dstChannelStepB)]);

    initVectors();
    process();

    registersPool.reset();
    this->postamble();
}

template <>
void GridSampleKernel<x64::avx512_core>::initVectors() {
    auto rAux = getReg64();
    Xbyak::Reg32 r32Aux(rAux.getIdx());

    if (jcp.dynamicShapes) {
        regChannelNum = getReg64();
        mov(regChannelNum, ptr[regParams + GET_OFF(channelsNum)]);
    }
    kTailMask = getMask();

    vSrcWidthF = getVmm();
    mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
    uni_vpbroadcastd(vSrcWidthF, ptr[rAux]);

    vSrcHeightF = getVmm();
    mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
    uni_vpbroadcastd(vSrcHeightF, ptr[rAux]);

    vZeros = getVmm();
    uni_vpxor(vZeros, vZeros, vZeros);

    if (one_of(jcp.interpolationMode, GridSampleInterpolationMode::BICUBIC, GridSampleInterpolationMode::BILINEAR)) {
        vOnesF = getVmm();
        mov(r32Aux, 0x3f800000); // 1.f
        vpbroadcastd(vOnesF, r32Aux);
    }

    if (jcp.alignCorners) {
        vWDenormCoefF = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(wDenormCoefF)]);
        uni_vpbroadcastd(vWDenormCoefF, ptr[rAux]);

        vHDenormCoefF = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(hDenormCoefF)]);
        uni_vpbroadcastd(vHDenormCoefF, ptr[rAux]);
    } else {
        vHalfF = getVmm();
        mov(r32Aux, 0x3f000000); // 0.5f
        vpbroadcastd(vHalfF, r32Aux);
    }

    static const unsigned gridPermMask[16]  = { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 };
    mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
    vGridPermMask = getVmm();
    uni_vmovups(vGridPermMask, ptr[rAux]);

    if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
        vDataTypeSizeB = getVmm();
        mov(rAux, dataTypeSize);
        vpbroadcastd(vDataTypeSizeB, r32Aux);
        vSrcWidthB = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcWidthB)]);
        uni_vpbroadcastd(vSrcWidthB, ptr[rAux]);
    } else if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
        vSrcHeightSub1F = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcHeightSub1F)]);
        uni_vpbroadcastd(vSrcHeightSub1F, ptr[rAux]);
        vSrcWidthSub1F = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcWidthSub1F)]);
        uni_vpbroadcastd(vSrcWidthSub1F, ptr[rAux]);
    } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
        vSrcHeightMul2F = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2F)]);
        uni_vpbroadcastd(vSrcHeightMul2F, ptr[rAux]);
        vSrcWidthMul2F = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2F)]);
        uni_vpbroadcastd(vSrcWidthMul2F, ptr[rAux]);
        vSrcHeightMul2Sub1F = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
        uni_vpbroadcastd(vSrcHeightMul2Sub1F, ptr[rAux]);
        vSrcWidthMul2Sub1F = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
        uni_vpbroadcastd(vSrcWidthMul2Sub1F, ptr[rAux]);
        if (jcp.alignCorners) {
            vAbsMask = getVmm();
            mov(r32Aux, 0x7fffffff);
            vpbroadcastd(vAbsMask, r32Aux);
        }
    }

    if (jcp.interpolationMode == GridSampleInterpolationMode::BICUBIC) {
        vConst_0_75 = getVmm();
        mov(r32Aux, 0xbf400000); // -0.75f
        vpbroadcastd(vConst_0_75, r32Aux);
        vConst_1_25 = getVmm();
        mov(r32Aux, 0x3fa00000); // 1.25f
        vpbroadcastd(vConst_1_25, r32Aux);
        vConst_1_50 = getVmm();
        mov(r32Aux, 0x3fc00000); // 1.5f
        vpbroadcastd(vConst_1_50, r32Aux);
        vConst_2_00 = getVmm();
        mov(r32Aux, 0x40000000); // 2.0f
        vpbroadcastd(vConst_2_00, r32Aux);
        vConst_2_25 = getVmm();
        mov(r32Aux, 0x40100000); // 2.25f
        vpbroadcastd(vConst_2_25, r32Aux);
    }
}

template <x64::cpu_isa_t isa> // Works for AVX2, AVX, SSE41
void GridSampleKernel<isa>::initVectors() {
    auto rAux = getReg64();

    vSrcWidthF = getVmm();
    mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
    uni_vmovups(vSrcWidthF, ptr[rAux]);

    if (one_of(jcp.interpolationMode, GridSampleInterpolationMode::BILINEAR, GridSampleInterpolationMode::NEAREST) ||
        (jcp.interpolationMode == GridSampleInterpolationMode::BICUBIC && (jcp.paddingMode == GridSamplePaddingMode::REFLECTION ||
                                                                          (jcp.paddingMode == GridSamplePaddingMode::BORDER && !jcp.alignCorners) ||
                                                                           jcp.paddingMode == GridSamplePaddingMode::ZEROS)) ) {
        vSrcHeightF = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
        uni_vmovups(vSrcHeightF, ptr[rAux]);
    }

    if (jcp.interpolationMode == GridSampleInterpolationMode::BICUBIC &&
        jcp.paddingMode == GridSamplePaddingMode::BORDER && jcp.alignCorners) {
        vHDenormCoefF = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(hDenormCoefF)]);
        uni_vmovups(vHDenormCoefF, ptr[rAux]);
    }

    if (jcp.interpolationMode != GridSampleInterpolationMode::BICUBIC) {
        if (one_of(jcp.paddingMode, GridSamplePaddingMode::BORDER, GridSamplePaddingMode::ZEROS) &&
            ((isa == x64::avx2 && jcp.interpolationMode == GridSampleInterpolationMode::NEAREST) || one_of(isa, x64::avx, x64::sse41))) {
            vZeros = getVmm();
            uni_vpxor(vZeros, vZeros, vZeros);
        }

        if (jcp.alignCorners) {
            mov(rAux, ptr[regParams + GET_OFF(wDenormCoefF)]);
            vWDenormCoefF = getVmm();
            uni_vmovups(vWDenormCoefF, ptr[rAux]);
            if (!(jcp.interpolationMode == GridSampleInterpolationMode::BILINEAR && jcp.paddingMode == GridSamplePaddingMode::ZEROS)) {
                mov(rAux, ptr[regParams + GET_OFF(hDenormCoefF)]);
                vHDenormCoefF = getVmm();
                uni_vmovups(vHDenormCoefF, ptr[rAux]);
            }
        } else {
            static const float halfArr[8] = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
            mov(rAux, reinterpret_cast<uintptr_t>(halfArr));
            vHalfF = getVmm();
            uni_vmovups(vHalfF, ptr[rAux]);
        }

        if (isa == x64::avx2 && jcp.interpolationMode == GridSampleInterpolationMode::NEAREST) {
            static const unsigned gridPermMask[8]  = { 0, 2, 4, 6, 1, 3, 5, 7 };
            mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
            vGridPermMask = getVmm();
            uni_vmovups(vGridPermMask, ptr[rAux]);
        }
    }

    if (jcp.interpolationMode == GridSampleInterpolationMode::BICUBIC ||
        (jcp.interpolationMode == GridSampleInterpolationMode::BILINEAR && jcp.paddingMode != GridSamplePaddingMode::ZEROS)) {
        static const float onesArr[8] = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
        mov(rAux, reinterpret_cast<uintptr_t>(onesArr));
        vOnesF = getVmm();
        uni_vmovups(vOnesF, ptr[rAux]);
    }
}

template <x64::cpu_isa_t isa> // Works for AVX512, AVX2, AVX, SSE41
void GridSampleKernel<isa>::process() {
    regWorkAmount = getReg64();

    // Batch loop
    Xbyak::Label lBatchLoop, lEnd;
    RegistersPool::Reg<Xbyak::Reg64> regBatch;

    for (uint64_t i = 0lu; i < jcp.batchNum; i++) {
        if (jcp.dynamicBatch) {
            regBatch = getReg64();
            mov(regBatch, ptr[regParams + GET_OFF(batchNum)]);

            L(lBatchLoop);
            cmp(regBatch, 0);
            jle(lEnd, T_NEAR);
        }

        mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);
        spatialLoop();

        if (jcp.dynamicShapes) {
            add(regSrc,  ptr[regParams + GET_OFF(srcBatchStepB)]);
        } else {
            add(regSrc, jcp.srcBatchStepB);
        }
        add(regGrid, ptr[regParams + GET_OFF(gridBatchStepB)]);
        add(regDst,  ptr[regParams + GET_OFF(dstBatchStepB)]);

        if (jcp.dynamicBatch) {
            dec(regBatch);
            jmp(lBatchLoop, T_NEAR);
            L(lEnd);
        }
    }
}

template <x64::cpu_isa_t isa> // Works for AVX512, AVX2, AVX, SSE41
void GridSampleKernel<isa>::spatialLoop() {
    auto vHCoord = getVmm();
    auto vWCoord = getVmm();

    Xbyak::Label lSpacialLoop, lTail;
    L(lSpacialLoop);
    {
        cmp(regWorkAmount, dataElPerVec);
        jl(lTail, T_NEAR);

        getCoordinates(vHCoord, vWCoord);
        denormalizeRawCoordinates(vWCoord, vHCoord);
        interpolation(vWCoord, vHCoord);

        sub(regWorkAmount, dataElPerVec);
        add(regDst, vlen);

        jmp(lSpacialLoop, T_NEAR);
    }

    L(lTail);
    vHCoord.release();
    vWCoord.release();
    tail();
}

template <x64::cpu_isa_t isa> // Works for AVX512, AVX2, AVX, SSE41
void GridSampleKernel<isa>::interpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    if (jcp.interpolationMode == GridSampleInterpolationMode::BILINEAR) {
        bilinearInterpolation(vWCoord, vHCoord, tail);
    } else if (jcp.interpolationMode == GridSampleInterpolationMode::BICUBIC) {
        bicubicInterpolation(vWCoord, vHCoord, tail);
    } else if (jcp.interpolationMode == GridSampleInterpolationMode::NEAREST) {
        nearestInterpolation(vWCoord, vHCoord, tail);
    }
}

template <x64::cpu_isa_t isa> // Works for AVX512, AVX2, AVX, SSE41
void GridSampleKernel<isa>::tail() {
    Xbyak::Label lEnd;
    cmp(regWorkAmount, 0);
    jle(lEnd, T_NEAR);

    auto vHCoord = getVmm();
    auto vWCoord = getVmm();

    getTailCoordinates(vHCoord, vWCoord);
    denormalizeRawCoordinates(vWCoord, vHCoord);
    interpolation(vWCoord, vHCoord, true);

    if (dataTypeSize > 1)
        sal(regWorkAmount, dataTypeShift); // Multiply by source data type size.
    add(regDst, regWorkAmount);

    L(lEnd);
}

template <>
void GridSampleKernel<x64::avx512_core>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    vpermd(vWCoord, vGridPermMask, ptr[regGrid]);      // Permute to XXXX.XXXX.YYYY.YYYY
    vshuff64x2(vHCoord, vWCoord, vHCoord, 0B11101110); // Extract Y component

    add(regGrid, vlen);

    auto vAux = getVmm();
    vpermd(vAux, vGridPermMask, ptr[regGrid]);         // Permute to XXXX.XXXX.YYYY.YYYY
    vshuff64x2(vWCoord, vWCoord, vAux, 0B01000100);    // Extract X component
    vshuff64x2(vHCoord, vHCoord, vAux, 0B11100100);    // Extract Y component

    add(regGrid, vlen);
}

template <>
void GridSampleKernel<x64::avx2>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    auto vAux = getVmm();
    Vmm vPermMask;
    RegistersPool::Reg<Vmm> permMaskHolder;

    if (vGridPermMask.isInitialized()) {
        vPermMask = vGridPermMask;
    } else {
        static const unsigned gridPermMask[8] = {0, 2, 4, 6, 1, 3, 5, 7};
        auto rAux = getReg64();
        permMaskHolder = getVmm();
        vPermMask = permMaskHolder;
        mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
        uni_vmovups(vPermMask, ptr[rAux]);
    }

    vpermd(vWCoord, vPermMask, ptr[regGrid]);          // Permute to XXXX.YYYY
    vperm2f128(vHCoord, vHCoord, vWCoord, 0B00000011); // Extract Y component

    add(regGrid, vlen);

    vpermd(vAux, vPermMask, ptr[regGrid]);             // Permute to XXXX.YYYY
    vperm2f128(vWCoord, vWCoord, vAux, 0B00100000);    // Extract X component
    vperm2f128(vHCoord, vHCoord, vAux, 0B00110000);    // Extract Y component

    add(regGrid, vlen);
}

template <x64::cpu_isa_t isa> // Works for AVX, SSE41
void GridSampleKernel<isa>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    auto vAux = getVmm();
    Xbyak::Xmm xmmWCoord(vWCoord.getIdx());
    Xbyak::Xmm xmmHCoord(vHCoord.getIdx());
    Xbyak::Xmm xmmAux(vAux.getIdx());
    const uint64_t xmmVlen = x64::cpu_isa_traits<x64::sse41>::vlen;

    uni_vmovups(xmmWCoord, ptr[regGrid]);
    uni_vpshufd(xmmWCoord, xmmWCoord, 0xD8);
    shufpd(xmmHCoord, xmmWCoord, 0x2);

    add(regGrid, xmmVlen);

    uni_vmovups(xmmAux, ptr[regGrid]);
    uni_vpshufd(xmmAux, xmmAux, 0xD8);
    shufpd(xmmWCoord, xmmAux, 0x0);
    shufpd(xmmHCoord, xmmAux, 0x3);

    add(regGrid, xmmVlen);

    if (isa == x64::avx) {
        Xbyak::Ymm ymmWCoord(vWCoord.getIdx());
        Xbyak::Ymm ymmHCoord(vHCoord.getIdx());

        vperm2f128(ymmWCoord, ymmWCoord, ymmWCoord, 0x1);
        vperm2f128(ymmHCoord, ymmHCoord, ymmHCoord, 0x1);

        // Here is movups + pshufd instead of vpshufd for two reasons:
        // 1. vpshufd zeroes the rest ov YMM.
        // 2. pshufd does not work with not aligned address.
        movups(xmmWCoord, ptr[regGrid]);
        pshufd(xmmWCoord, xmmWCoord, 0xD8);
        shufpd(xmmHCoord, xmmWCoord, 0x2);

        add(regGrid, xmmVlen);

        uni_vpshufd(xmmAux, ptr[regGrid], 0xD8);
        shufpd(xmmWCoord, xmmAux, 0x0);
        shufpd(xmmHCoord, xmmAux, 0x3);

        add(regGrid, xmmVlen);

        vperm2f128(ymmWCoord, ymmWCoord, ymmWCoord, 0x1);
        vperm2f128(ymmHCoord, ymmHCoord, ymmHCoord, 0x1);
    }
}

template <>
void GridSampleKernel<x64::avx512_core>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lEnd, lGridShift, lRest;

    auto vAux = getVmm();
    auto rAux = getReg64();

    mov(rAux, regWorkAmount);
    sal(rAux, 0x1); // Multiply by gridShape[3].
    cmp(regWorkAmount, dataElPerVec / 2);
    jl(lRest, T_NEAR);
    {
        vpermd(vWCoord, vGridPermMask, ptr[regGrid]);
        vshuff64x2(vHCoord, vWCoord, vHCoord, 0B11101110); // Extract Y component

        add(regGrid, vlen);
        sub(rAux, dataElPerVec);
        cmp(rAux, 0);
        jle(lEnd, T_NEAR);

        fillRestWorkMask(kTailMask, rAux);
        uni_vmovups((Vmm)vAux | kTailMask, ptr[regGrid]);
        vpermd(vAux, vGridPermMask, vAux);
        Xbyak::Ymm ymmAux(vAux.getIdx());
        vshuff64x2(vWCoord, vWCoord, vAux, 0B01000100);    // Extract X component
        vshuff64x2(vHCoord, vHCoord, vAux, 0B11100100);    // Extract Y component

        jmp(lGridShift, T_NEAR);
    }
    L(lRest);
    {
        fillRestWorkMask(kTailMask, rAux);
        uni_vmovups(vWCoord | kTailMask, ptr[regGrid]);
        vpermd(vWCoord, vGridPermMask, vWCoord);
        vshuff64x2(vHCoord, vWCoord, vHCoord, 0B11101110); // Extract Y component
    }

    L(lGridShift);
    if (dataTypeSize > 1)
        sal(rAux, dataTypeShift); // Multiply by source data type size.
    add(regGrid, rAux);

    L(lEnd);

    fillRestWorkMask(kTailMask, regWorkAmount);
}

template <>
void GridSampleKernel<x64::avx2>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lRest, lGridShift, lEnd;

    auto rAux = getReg64();
    Vmm vPermMask;
    RegistersPool::Reg<Vmm> permMaskHolder;

    if (vGridPermMask.isInitialized()) {
        vPermMask = vGridPermMask;
    } else {
        static const unsigned gridPermMask[8] = {0, 2, 4, 6, 1, 3, 5, 7};
        permMaskHolder = getVmm();
        vPermMask = permMaskHolder;
        mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
        uni_vmovups(vPermMask, ptr[rAux]);
    }

    mov(rAux, regWorkAmount);
    sal(rAux, 0x1); // multiply by gridShape[3] == 2
    cmp(regWorkAmount, dataElPerVec / 2);
    jl(lRest, T_NEAR);
    {
        vpermd(vWCoord, vPermMask, ptr[regGrid]);          // Permute to XXXX.YYYY
        vperm2f128(vHCoord, vHCoord, vWCoord, 0B00000011); // Extract Y component

        add(regGrid, vlen);
        sub(rAux, dataElPerVec);
        cmp(rAux, 0);
        jle(lEnd, T_NEAR);

        auto vAux  = getVmm();
        load(vAux, ptr[regGrid], rAux, dataTypeSize);
        vpermd(vAux, vPermMask, vAux);
        vperm2f128(vWCoord, vWCoord, vAux, 0B00100000); // Extract X component
        vperm2f128(vHCoord, vHCoord, vAux, 0B00110000); // Extract Y component

        jmp(lGridShift, T_NEAR);
    }
    L(lRest);
    {
        load(vWCoord, ptr[regGrid], rAux, dataTypeSize);
        vpermd(vWCoord, vPermMask, vWCoord);               // Permute to XXXX.YYYY
        vperm2f128(vHCoord, vHCoord, vWCoord, 0B00000011); // Extract Y component
    }

    L(lGridShift);
    if (dataTypeSize > 1)
        sal(rAux, dataTypeShift); // Multiply by source data type size.
    add(regGrid, rAux);

    L(lEnd);
}

template <>
void GridSampleKernel<x64::avx>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lLoop2End, lEnd;

    Xbyak::Xmm xmmWCoord(vWCoord.getIdx());
    Xbyak::Xmm xmmHCoord(vHCoord.getIdx());

    auto rGridRest = getReg64();
    mov(rGridRest, regWorkAmount);
    sal(rGridRest, 0x1); // multiply by gridShape[3] == 2

    for (size_t i = 0; i < dataElPerVec; i++) {
        cmp(rGridRest, 0);
        jle(lEnd, T_NEAR);

        if (gridTypeSize == 4)
            pinsrd(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);
        else if (gridTypeSize == 2)
            pinsrw(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);

        add(regGrid, gridTypeSize);
        dec(rGridRest);
    }

    cmp(rGridRest, 0);
    jle(lEnd, T_NEAR);

    vperm2f128(vWCoord, vWCoord, vWCoord, 0x1);
    vperm2f128(vHCoord, vHCoord, vHCoord, 0x1);

    for (size_t i = 0; i < dataElPerVec; i++) {
        cmp(rGridRest, 0);
        jle(lLoop2End, T_NEAR);

        if (gridTypeSize == 4)
            pinsrd(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);
        else if (gridTypeSize == 2)
            pinsrw(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);

        add(regGrid, gridTypeSize);
        dec(rGridRest);
    }

    L(lLoop2End);
    vperm2f128(vWCoord, vWCoord, vWCoord, 0x1);
    vperm2f128(vHCoord, vHCoord, vHCoord, 0x1);

    L(lEnd);
}

template <>
void GridSampleKernel<x64::sse41>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lRest, lHShuf, lGridShift, lEnd;
    auto rAux = getReg64();

    mov(rAux, regWorkAmount);
    sal(rAux, 0x1); // Multiply by gridShape[3] == 2
    cmp(regWorkAmount, dataElPerVec / 2);
    jl(lRest, T_NEAR);
    {
        // Here is movups + pshufd instead of pshufd due to
        // pshufd does not work with not aligned address.
        movups(vWCoord, ptr[regGrid]);
        pshufd(vWCoord, vWCoord, 0B11011000);
        shufpd(vHCoord, vWCoord, 0B00000010);

        add(regGrid, vlen);
        sub(rAux, dataElPerVec);
        cmp(rAux, 0);
        jle(lHShuf, T_NEAR);

        auto vAux = getVmm();
        load(vAux, ptr[regGrid], rAux, dataTypeSize);
        pshufd(vAux, vAux, 0B11011000);
        shufpd(vWCoord, vAux, 0x0);        // Extract X component
        shufpd(vHCoord, vAux, 0B00000011); // Extract Y component

        jmp(lGridShift, T_NEAR);
        L(lHShuf);
        shufpd(vHCoord, vHCoord, 0B00000001); // Extract Y component
        jmp(lEnd, T_NEAR);
    }
    L(lRest);
    {
        load(vWCoord, ptr[regGrid], rAux, dataTypeSize);
        pshufd(vWCoord, vWCoord, 0B11011000); // Extract X component
        shufpd(vHCoord, vWCoord, 0B00000010); // Extract Y component
        shufpd(vHCoord, vHCoord, 0B00000001);
    }

    L(lGridShift);
    if (dataTypeSize > 1)
        sal(rAux, dataTypeShift); // Multiply by source data type size.
    add(regGrid, rAux);

    L(lEnd);
}

template <x64::cpu_isa_t isa> // Works for AVX512, AVX2, AVX, SSE41
void GridSampleKernel<isa>::denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord) {
    if (jcp.alignCorners) {
        if (vWDenormCoefF.isInitialized()) {
            uni_vfmadd132ps(vWCoord, vWDenormCoefF, vWDenormCoefF);
        } else {
            auto rAux = getReg64();
            auto vAux = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(wDenormCoefF)]);
            uni_vmovups(vAux, ptr[rAux]);
            uni_vfmadd132ps(vWCoord, vAux, vAux);
        }

        if (vHDenormCoefF.isInitialized()) {
            uni_vfmadd132ps(vHCoord, vHDenormCoefF, vHDenormCoefF);
        } else {
            auto rAux = getReg64();
            auto vAux = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(hDenormCoefF)]);
            uni_vmovups(vAux, ptr[rAux]);
            uni_vfmadd132ps(vHCoord, vAux, vAux);
        }
    } else {
        Vmm vHalfTmp;
        RegistersPool::Reg<Vmm> halfHolder;
        if (vHalfF.isInitialized()) {
            vHalfTmp = vHalfF;
        } else {
            auto rAux = getReg64();
            halfHolder = getVmm();
            vHalfTmp = halfHolder;
            static const float halfValues[x64::cpu_isa_traits<x64::avx512_core>::vlen / sizeof(float)] =
                    { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
            mov(rAux, reinterpret_cast<uintptr_t>(halfValues));
            uni_vmovups(vHalfTmp, ptr[rAux]);
        }

        if (vSrcWidthF.isInitialized()) {
            uni_vfmadd132ps(vWCoord, vSrcWidthF, vSrcWidthF);
        } else {
            auto rAux = getReg64();
            auto vAux = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
            uni_vpbroadcastd(vAux, ptr[rAux]);
            uni_vfmadd132ps(vWCoord, vAux, vAux);
        }
        uni_vfmsub132ps(vWCoord, vHalfTmp, vHalfTmp);

        if (vSrcHeightF.isInitialized()) {
            uni_vfmadd132ps(vHCoord, vSrcHeightF, vSrcHeightF);
        } else {
            auto rAux = getReg64();
            auto vAux = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
            uni_vpbroadcastd(vAux, ptr[rAux]);
            uni_vfmadd132ps(vHCoord, vAux, vAux);
        }
        uni_vfmsub132ps(vHCoord, vHalfTmp, vHalfTmp);
    }
}

template <>
void GridSampleKernel<x64::avx512_core>::zerosPaddingW(const Vmask& kDst, const Vmm& vCoord) {
    vcmpps(kDst, vCoord, vSrcWidthF, CMP_LT_PS);    // vCoord < vUpperBound
    vcmpps(kDst | kDst, vZeros, vCoord, CMP_LE_PS); // vCoord >= vZeros
}

template <>
void GridSampleKernel<x64::avx512_core>::zerosPaddingH(const Vmask& kDst, const Vmm& vCoord, const Vmask& kMaskW) {
    vcmpps(kDst | kMaskW, vCoord, vSrcHeightF, CMP_LT_PS); // vCoord < vUpperBound
    vcmpps(kDst | kDst, vZeros, vCoord, CMP_LE_PS);        // vCoord >= vZeros
}

template <>
void GridSampleKernel<x64::avx512_core>::zerosPadding(const Vmask& kDst, const Vmm& vHCoord, const Vmm& vWCoord) {
    zerosPaddingW(kDst, vWCoord);
    zerosPaddingH(kDst, vHCoord, kDst);
}

template <>
void GridSampleKernel<x64::sse41>::zerosPaddingW(const Vmask& kDst, const Vmm& vWCoord) {
    auto vAux = getVmm();

    if (vSrcWidthF.isInitialized()) {
        uni_vcmpps(vAux, vWCoord, vSrcWidthF, CMP_LT_PS); // vWCoord < vSrcWidthF
    } else {
        auto rAux = getReg64();
        mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
        uni_vcmpps(vAux, vWCoord, ptr[rAux], CMP_LT_PS);  // vWCoord < vSrcWidthF
    }

    uni_vpxor(kDst, kDst, kDst);
    uni_vcmpps(kDst, kDst, vWCoord, CMP_LE_PS);           // vWCoord >= vZeros
    uni_vpand(kDst, kDst, vAux);                    // vZeros <= vWCoord < vSrcWidthF
}

template <>
void GridSampleKernel<x64::sse41>::zerosPaddingH(const Vmask& kDst, const Vmm& vHCoord, const Vmask& kMaskW) {
    auto vAux = getVmm();

    if (vSrcHeightF.isInitialized()) {
        uni_vcmpps(vAux, vHCoord, vSrcHeightF, CMP_LT_PS); // vHCoord < vSrcHeightF
    } else {
        auto rAux = getReg64();
        mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
        uni_vcmpps(vAux, vHCoord, ptr[rAux], CMP_LT_PS);   // vHCoord < vSrcHeightF
    }

    uni_vmovups(kDst, kMaskW);
    uni_vpand(kDst, kDst, vAux); // vHCoord < vSrcHeightF && vZeros <= vWCoord < vSrcWidthF
    uni_vpxor(vAux, vAux, vAux);
    uni_vcmpps(vAux, vAux, vHCoord, CMP_LE_PS); // vHCoord >= vZeros
    uni_vpand(kDst, kDst, vAux); // vZeros <= vHCoord < vSrcHeightF && vZeros <= vWCoord < vSrcWidthF
}

template <>
void GridSampleKernel<x64::sse41>::zerosPadding(const Vmask& kDst, const Vmm& vHCoord, const Vmm& vWCoord) {
    zerosPaddingW(kDst, vWCoord);
    zerosPaddingH(kDst, vHCoord, kDst);
}

template <x64::cpu_isa_t isa> // Works for AVX2, AVX
void GridSampleKernel<isa>::zerosPaddingW(const Vmask& kDst, const Vmm& vCoord) {
    auto vAux = getVmm();
    Vmm vZerosTmp;
    RegistersPool::Reg<Vmm> zerosHolder;
    if (vZeros.isInitialized()) {
        vZerosTmp = vZeros;
    } else {
        zerosHolder = getVmm();
        vZerosTmp = zerosHolder;
        uni_vpxor(vZerosTmp, vZerosTmp, vZerosTmp);
    }

    if (vSrcWidthF.isInitialized()) {
        uni_vcmpps(vAux, vCoord, vSrcWidthF, CMP_LT_PS); // vWCoord < vSrcWidthF
    } else {
        auto rAux = getReg64();
        mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
        uni_vcmpps(vAux, vCoord, ptr[rAux], CMP_LT_PS);  // vWCoord < vSrcWidthF
    }

    uni_vcmpps(kDst, vZerosTmp, vCoord, CMP_LE_PS);      // vWCoord >= vZeros
    uni_vandps(kDst, kDst, vAux);                  // vZeros <= vWCoord < vSrcWidthF
}

template <x64::cpu_isa_t isa> // Works for AVX2, AVX
void GridSampleKernel<isa>::zerosPaddingH(const Vmask& kDst, const Vmm& vCoord, const Vmask& kMaskW) {
    auto vAux = getVmm();
    Vmm vZerosTmp;
    RegistersPool::Reg<Vmm> zerosHolder;
    if (vZeros.isInitialized()) {
        vZerosTmp = vZeros;
    } else {
        zerosHolder = getVmm();
        vZerosTmp = zerosHolder;
        uni_vpxor(vZerosTmp, vZerosTmp, vZerosTmp);
    }

    if (vSrcHeightF.isInitialized()) {
        uni_vcmpps(vAux, vCoord, vSrcHeightF, CMP_LT_PS); // vHCoord < vSrcHeightF
    } else {
        auto rAux = getReg64();
        mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
        uni_vcmpps(vAux, vCoord, ptr[rAux], CMP_LT_PS);   // vHCoord < vSrcHeightF
    }

    uni_vandps(kDst, kMaskW, vAux);
    uni_vcmpps(vAux, vZerosTmp, vCoord, CMP_LE_PS);       // vHCoord >= vZeros
    uni_vandps(kDst, kDst, vAux);
}

template <x64::cpu_isa_t isa> // Works for AVX2, AVX
void GridSampleKernel<isa>::zerosPadding(const Vmask& kDst, const Vmm& vHCoord, const Vmm& vWCoord) {
    bool releaseZeroVec = false;
    if (!vZeros.isInitialized()) {
        releaseZeroVec = true;
        vZeros = getVmm();
        uni_vpxor(vZeros, vZeros, vZeros);
    }

    zerosPaddingW(kDst, vWCoord);
    zerosPaddingH(kDst, vHCoord, kDst);

    if (releaseZeroVec) {
        vZeros.release();
    }
}

template <>
void GridSampleKernel<x64::avx512_core>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
    vrangeps(vCoordDst, vCoordOrigin, dim == coord::w ? vSrcWidthSub1F : vSrcHeightSub1F, 0x0); // vWCoord >= vSrcWidthF
    vrangeps(vCoordDst, vCoordDst, vZeros, 0x1); // vWCoord < vZeros
}

template <x64::cpu_isa_t isa> // Works for AVX2, AVX, SSE41
void GridSampleKernel<isa>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
    auto rAux = getReg64();
    auto vAux = getVmm();
    RegistersPool::Reg<Vmm> vAux1;

    Vmm vSub1F;
    if (dim == coord::w) {
        if (vSrcWidthSub1F.isInitialized()) {
            vSub1F = vSrcWidthSub1F;
        } else {
            vAux1 = getVmm();
            vSub1F = vAux1;
            mov(rAux, ptr[regParams + GET_OFF(srcWidthSub1F)]);
            uni_vmovups(vSub1F, ptr[rAux]);
        }
    } else if (dim == coord::h) {
        if (vSrcHeightSub1F.isInitialized()) {
            vSub1F = vSrcHeightSub1F;
        } else {
            vAux1 = getVmm();
            vSub1F = vAux1;
            mov(rAux, ptr[regParams + GET_OFF(srcHeightSub1F)]);
            uni_vmovups(vSub1F, ptr[rAux]);
        }
    }

    uni_vcmpps(vAux, vCoordOrigin, vSub1F, CMP_LE_PS);  // vCoord <= vUpperBound
    uni_vandps(vCoordDst, vCoordOrigin, vAux);
    uni_vandnps(vAux, vAux, vSub1F);
    uni_vaddps(vCoordDst, vCoordDst, vAux);

    if (vZeros.isInitialized()) {
        uni_vcmpps(vAux, vCoordDst, vZeros, 0x6); // vCoord >= vZeros
    } else {
        if (isa == x64::sse41) {
            if (!vAux1.isInitialized()) {
                vAux1 = getVmm();
                vSub1F = vAux1;
            }
            uni_vpxor(vSub1F, vSub1F, vSub1F);
            uni_vcmpps(vAux, vCoordDst, vSub1F, 0x6); // vCoord >= vZeros
        } else {
            uni_vpxor(vAux, vAux, vAux);
            uni_vcmpps(vAux, vCoordDst, vAux, 0x6);   // vCoord >= vZeros
        }
    }
    uni_vandps(vCoordDst, vCoordDst, vAux);
}

template <>
void GridSampleKernel<x64::avx512_core>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
    auto vAux = getVmm();
    auto kAux = getMask();
    const auto& vSrcDimMul2Sub1F = dim == coord::w ? vSrcWidthMul2Sub1F : vSrcHeightMul2Sub1F;

    if (jcp.alignCorners) {
        // abs(x) % D21
        uni_vandps(vCoordDst, vCoordOrigin, vAbsMask); // abs(x)
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2Sub1F);
        uni_vroundps(vAux, vAux, 0x3);                       // Truncation
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2Sub1F); // abs(x) % D21

        // Check that the result does not exceed the divisor.
        vcmpps(kAux, vSrcDimMul2Sub1F, vCoordDst, CMP_LE_PS);
        uni_vmovups(vCoordDst | kAux, vZeros);
        vrangeps(vCoordDst, vCoordDst, vZeros, 0x1);
    } else {
        const auto& vSrcDimMul2F = dim == coord::w ? vSrcWidthMul2F : vSrcHeightMul2F;
        // (x % D2 + D2) % D2
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2F);
        uni_vroundps(vAux, vAux, 0x3);                   // Truncation
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2F); // x % D2
        uni_vaddps(vCoordDst, vCoordDst, vSrcDimMul2F);  // x % D2 + D2
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2F);
        uni_vroundps(vAux, vAux, 0x3);                   // Truncation
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2F); // (x % D2 + D2) % D2

        // Check that the result does not exceed the divisor.
        vcmpps(kAux, vSrcDimMul2F, vCoordDst, CMP_LE_PS);
        uni_vmovups(vCoordDst | kAux, vZeros);
        vrangeps(vCoordDst, vCoordDst, vZeros, 0x1);
    }

    uni_vsubps(vAux, vSrcDimMul2Sub1F, vCoordDst);
    vcmpps(kAux, dim == coord::w ? vSrcWidthF : vSrcHeightF, vCoordDst, CMP_LE_PS); // vCoordDst >= vSrcDimF
    uni_vmovups(vCoordDst | kAux, vAux);
}

template <x64::cpu_isa_t isa> // Works for AVX2, AVX, SSE41
void GridSampleKernel<isa>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
    auto rAux  = getReg64();
    auto vAux0 = getVmm();
    auto vAux1 = getVmm();

    // D2  = Dim * 2
    // D21 = (Dim - 1) * 2
    if (jcp.alignCorners) {
        // x' = abs(x) % D21 - D21
        static const unsigned absMask[8] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
        if (isa ==x64::sse41) {
            static const unsigned *absPtr = absMask + (reinterpret_cast<int64_t>(absMask) % 16) / sizeof(unsigned);
            mov(rAux, reinterpret_cast<uintptr_t>(absPtr));
        } else {
            mov(rAux, reinterpret_cast<uintptr_t>(absMask));
        }
        uni_vandps(vCoordDst, vCoordOrigin, ptr[rAux]); // abs(x)

        Vmm vMul2Sub1;
        if (dim == coord::w) {
            if (vSrcWidthMul2Sub1F.isInitialized()) {
                vMul2Sub1 = vSrcWidthMul2Sub1F;
            } else {
                vMul2Sub1 = vAux1;
                mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
                uni_vmovups(vAux1, ptr[rAux]);
            }
        } else if (dim == coord::h) {
            if (vSrcHeightMul2Sub1F.isInitialized()) {
                vMul2Sub1 = vSrcHeightMul2Sub1F;
            } else {
                vMul2Sub1 = vAux1;
                mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
                uni_vmovups(vAux1, ptr[rAux]);
            }
        }
        uni_vdivps(vAux0, vCoordDst, vMul2Sub1);
        uni_vroundps(vAux0, vAux0, 0x3);               // Truncation
        uni_vfnmadd231ps(vCoordDst, vAux0, vMul2Sub1); // abs(x) % D21

        // Check that the result does not exceed the divisor.
        uni_vcmpps(vAux0, vCoordDst, vMul2Sub1, CMP_LT_PS);
        uni_vandps(vCoordDst, vCoordDst, vAux0);
        uni_vxorps(vAux0, vAux0, vAux0);
        uni_vcmpps(vAux0, vAux0, vCoordDst, CMP_LE_PS);
        uni_vandps(vCoordDst, vCoordDst, vAux0);

        uni_vsubps(vAux0, vCoordDst, vMul2Sub1);       // abs(x) % D21 - D21
    } else {
        // x' = (x % D2 + D2) % D2 - D21
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        Vmm vMul2;
        if (dim == coord::w) {
            if (vSrcWidthMul2F.isInitialized()) {
                vMul2 = vSrcWidthMul2F;
            } else {
                vMul2 = vAux1;
                mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2F)]);
                uni_vmovups(vAux1, ptr[rAux]);
            }
        } else if (dim == coord::h) {
            if (vSrcHeightMul2F.isInitialized()) {
                vMul2 = vSrcHeightMul2F;
            } else {
                vMul2 = vAux1;
                mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2F)]);
                uni_vmovups(vAux1, ptr[rAux]);
            }
        }
        uni_vdivps(vAux0, vCoordOrigin, vMul2);
        uni_vroundps(vAux0, vAux0, 0x3);           // Truncation
        uni_vfnmadd231ps(vCoordDst, vAux0, vMul2); // x % D2
        uni_vaddps(vCoordDst, vCoordDst, vMul2);   // x % D2 + D2
        uni_vdivps(vAux0, vCoordDst, vMul2);
        uni_vroundps(vAux0, vAux0, 0x3);           // Truncation
        uni_vfnmadd231ps(vCoordDst, vAux0, vMul2); // (x % D2 + D2) % D2

        // Check that the result does not exceed the divisor.
        uni_vcmpps(vAux0, vCoordDst, vMul2, CMP_LT_PS);
        uni_vandps(vCoordDst, vCoordDst, vAux0);
        uni_vxorps(vAux0, vAux0, vAux0);
        uni_vcmpps(vAux0, vAux0, vCoordDst, CMP_LE_PS);
        uni_vandps(vCoordDst, vCoordDst, vAux0);

        if (dim == coord::w) {
            if (vSrcWidthMul2Sub1F.isInitialized()) {
                uni_vsubps(vAux0, vCoordDst, vSrcWidthMul2Sub1F);
            } else {
                mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
                uni_vsubps(vAux0, vCoordDst, ptr[rAux]);
            }
        } else if (dim == coord::h) {
            if (vSrcHeightMul2Sub1F.isInitialized()) {
                uni_vsubps(vAux0, vCoordDst, vSrcHeightMul2Sub1F);
            } else {
                mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
                uni_vsubps(vAux0, vCoordDst, ptr[rAux]);
            }
        }
    }

    if (dim == coord::w) {
        if (vSrcWidthF.isInitialized()) {
            uni_vcmpps(vAux1, vCoordDst, vSrcWidthF, CMP_LT_PS);  // vCoordDst < vUpperBound
        } else {
            mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
            uni_vcmpps(vAux1, vCoordDst, ptr[rAux], CMP_LT_PS);   // vCoordDst < vUpperBound
        }
    } else {
        if (vSrcHeightF.isInitialized()) {
            uni_vcmpps(vAux1, vCoordDst, vSrcHeightF, CMP_LT_PS); // vCoordDst < vUpperBound
        } else {
            mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
            uni_vcmpps(vAux1, vCoordDst, ptr[rAux], CMP_LT_PS);   // vCoordDst < vUpperBound
        }
    }

    uni_vandps(vCoordDst, vCoordDst, vAux1);
    uni_vandnps(vAux1, vAux1, vAux0);
    uni_vsubps(vCoordDst, vCoordDst, vAux1); // set -x' for vCoordDst >= Dim
}

template <>
void GridSampleKernel<x64::avx512_core>::bicubicCoefficients(const Vmm& vCoef, const Vmm& vDDim, const uint8_t idx) {
    if (idx == 0) {
        uni_vmovups(vCoef, vDDim);
        uni_vfnmadd132ps(vCoef, vOnesF, vConst_2_00);
        uni_vfmadd231ps(vCoef, vDDim, vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vmulps(vCoef, vCoef, vConst_0_75);
    } else if (idx == 1) {
        uni_vmovups(vCoef, vDDim);
        vfmsub132ps(vCoef, vConst_2_25, vConst_1_25);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vfmadd132ps(vCoef, vOnesF, vDDim);
    } else if (idx == 2) {
        uni_vmovups(vCoef, vDDim);
        uni_vfnmadd132ps(vCoef, vConst_1_50, vConst_1_25);
        uni_vfmsub132ps(vCoef, vConst_0_75, vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
    } else if (idx == 3) {
        uni_vmulps(vCoef, vConst_0_75, vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vfnmadd132ps(vCoef, vCoef, vDDim);
    }
}

template <>
void GridSampleKernel<x64::avx2>::bicubicCoefficients(const Vmm& vCoef, const Vmm& vDDim, const uint8_t idx) {
    static const size_t elPerVec = x64::cpu_isa_traits<x64::avx2>::vlen / sizeof(float);;
    static const float const_0_75[elPerVec] = { -0.75f, -0.75f, -0.75f, -0.75f, -0.75f, -0.75f, -0.75f, -0.75f };
    static const float const_1_25[elPerVec] = { 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f };
    static const float const_1_50[elPerVec] = { 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f };
    static const float const_2_00[elPerVec] = { 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f };
    static const float const_2_25[elPerVec] = { 2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f };

    auto rAux = getReg64();

    if (idx == 0) {
        uni_vmovups(vCoef, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_2_00));
        uni_vfnmadd132ps(vCoef, vOnesF, ptr[rAux]);
        uni_vfmadd231ps(vCoef, vDDim, vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
        uni_vmulps(vCoef, vCoef, ptr[rAux]);
    } else if (idx == 1) {
        mov(rAux, reinterpret_cast<uintptr_t>(const_1_25));
        uni_vmulps(vCoef, vDDim, ptr[rAux]);
        mov(rAux, reinterpret_cast<uintptr_t>(const_2_25));
        uni_vsubps(vCoef, vCoef, ptr[rAux]);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vfmadd132ps(vCoef, vOnesF, vDDim);
    } else if (idx == 2) {
        mov(rAux, reinterpret_cast<uintptr_t>(const_1_25));
        uni_vmulps(vCoef, vDDim, ptr[rAux]);
        mov(rAux, reinterpret_cast<uintptr_t>(const_1_50));
        uni_vsubps(vCoef, vCoef, ptr[rAux]);
        mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
        vfnmsub213ps(vCoef, vDDim, ptr[rAux]);
        uni_vmulps(vCoef, vCoef, vDDim);
    } else if (idx == 3) {
        mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
        uni_vmulps(vCoef, vDDim, ptr[rAux]);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vfnmadd132ps(vCoef, vCoef, vDDim);
    }
}

template <>
void GridSampleKernel<x64::avx>::bicubicCoefficients(const Vmm& vCoef, const Vmm& vDDim, const uint8_t idx) {
    static const size_t elPerVec = x64::cpu_isa_traits<x64::avx>::vlen / sizeof(float);
    static const float const_0_75[elPerVec] = { -0.75f, -0.75f, -0.75f, -0.75f, -0.75f, -0.75f, -0.75f, -0.75f };
    static const float const_1_25[elPerVec] = { 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f };
    static const float const_1_50[elPerVec] = { 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f };
    static const float const_2_00[elPerVec] = { 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f };
    static const float const_2_25[elPerVec] = { 2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f };

    auto rAux = getReg64();
    auto vAux = getVmm();

    if (idx == 0) {
        uni_vmovups(vCoef, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_2_00));
        uni_vfnmadd132ps(vCoef, vOnesF, ptr[rAux]);
        uni_vmulps(vAux, vDDim, vDDim);
        uni_vaddps(vCoef, vCoef, vAux);
        uni_vmulps(vCoef, vCoef, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
        uni_vmulps(vCoef, vCoef, ptr[rAux]);
    } else if (idx == 1) {
        mov(rAux, reinterpret_cast<uintptr_t>(const_1_25));
        uni_vmulps(vCoef, vDDim, ptr[rAux]);
        mov(rAux, reinterpret_cast<uintptr_t>(const_2_25));
        uni_vsubps(vCoef, vCoef, ptr[rAux]);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vfmadd132ps(vCoef, vOnesF, vDDim);
    } else if (idx == 2) {
        mov(rAux, reinterpret_cast<uintptr_t>(const_1_25));
        uni_vmulps(vAux, vDDim, ptr[rAux]);
        mov(rAux, reinterpret_cast<uintptr_t>(const_1_50));
        uni_vmovups(vCoef, ptr[rAux]);
        uni_vsubps(vCoef, vCoef, vAux);
        uni_vmulps(vCoef, vCoef, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
        uni_vsubps(vCoef, vCoef, ptr[rAux]);
        uni_vmulps(vCoef, vCoef, vDDim);
    } else if (idx == 3) {
        mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
        uni_vmulps(vCoef, vDDim, ptr[rAux]);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vmulps(vAux, vCoef, vDDim);
        uni_vsubps(vCoef, vCoef, vAux);
    }
}

template <>
void GridSampleKernel<x64::sse41>::bicubicCoefficients(const Vmm& vCoef, const Vmm& vDDim, const uint8_t idx) {
    static const size_t elToAllocate = 2 * x64::cpu_isa_traits<x64::sse41>::vlen / sizeof(float);
    // Allocation with a margin for address alignment.
    static const float c_0_75[elToAllocate] = { -0.75f, -0.75f, -0.75f, -0.75f, -0.75f, -0.75f, -0.75f, -0.75f };
    static const float c_1_25[elToAllocate] = { 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f };
    static const float c_1_50[elToAllocate] = { 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f };
    static const float c_2_00[elToAllocate] = { 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f };
    static const float c_2_25[elToAllocate] = { 2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f };
    // Address alignment for XMM.
    static const float* const_0_75 = c_0_75 + (reinterpret_cast<int64_t>(c_0_75) % 16) / sizeof(float);
    static const float* const_1_25 = c_1_25 + (reinterpret_cast<int64_t>(c_1_25) % 16) / sizeof(float);
    static const float* const_1_50 = c_1_50 + (reinterpret_cast<int64_t>(c_1_50) % 16) / sizeof(float);
    static const float* const_2_00 = c_2_00 + (reinterpret_cast<int64_t>(c_2_00) % 16) / sizeof(float);
    static const float* const_2_25 = c_2_25 + (reinterpret_cast<int64_t>(c_2_25) % 16) / sizeof(float);

    auto rAux = getReg64();
    auto vAux = getVmm();

    if (idx == 0) {
        uni_vmovups(vAux, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_2_00));
        uni_vmulps(vAux, vAux, ptr[rAux]);
        uni_vsubps(vAux, vAux, vOnesF);
        uni_vmovups(vCoef, vDDim);
        uni_vmulps(vCoef, vCoef, vCoef);
        uni_vsubps(vCoef, vCoef, vAux);
        uni_vmulps(vCoef, vCoef, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
        uni_vmulps(vCoef, vCoef, ptr[rAux]);
    } else if (idx == 1) {
        uni_vmovups(vCoef, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_1_25));
        uni_vmulps(vCoef, vCoef, ptr[rAux]);
        mov(rAux, reinterpret_cast<uintptr_t>(const_2_25));
        uni_vsubps(vCoef, vCoef, ptr[rAux]);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vfmadd132ps(vCoef, vOnesF, vDDim);
    } else if (idx == 2) {
        uni_vmovups(vAux, vDDim);
        uni_vmulps(vAux, vDDim, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_1_25));
        uni_vmulps(vAux, vAux, ptr[rAux]);
        mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
        uni_vaddps(vAux, vAux, ptr[rAux]);
        uni_vmovups(vCoef, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_1_50));
        uni_vmulps(vCoef, vCoef, ptr[rAux]);
        uni_vsubps(vCoef, vCoef, vAux);
        uni_vmulps(vCoef, vCoef, vDDim);
    } else if (idx == 3) {
        uni_vmovups(vCoef, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
        uni_vmulps(vCoef, vCoef, ptr[rAux]);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vmovups(vAux, vCoef);
        uni_vmulps(vAux, vAux, vDDim);
        uni_vsubps(vCoef, vCoef, vAux);
    }
}

template <x64::cpu_isa_t isa> // Works for AVX512, AVX2, AVX, SSE41
void GridSampleKernel<isa>::nearestInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    const auto& vSrcShift = vWCoord;
    const auto& vAux      = vHCoord;
    auto kGatherMask      = getMask();
    auto kAuxMask         = getMask();

    uni_vroundps(vWCoord, vWCoord, 0x0); // Round near
    uni_vroundps(vHCoord, vHCoord, 0x0); // Round near

    bool useMask = false, zeroFill = false;
    if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
        useMask = zeroFill = true;
        zerosPadding(kGatherMask, vHCoord, vWCoord);
    } else if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
        borderPadding(vWCoord, vWCoord, coord::w);
        borderPadding(vHCoord, vHCoord, coord::h);
    } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
        reflectionPadding(vWCoord, vWCoord, coord::w);
        reflectionPadding(vHCoord, vHCoord, coord::h);
    }

    hwShiftPs2dq(vSrcShift, vHCoord, vWCoord, vSrcWidthF);

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    RegistersPool::Reg<Xbyak::Reg64> rChannel;
    auto rSrcTmp = getReg64();
    auto rDstTmp = getReg64();
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);

    for (uint64_t ch = 0; ch < jcp.cannelNum; ch++) {
        if (jcp.dynamicChannel) {
            rChannel = getReg64();
            mov(rChannel, ptr[regParams + GET_OFF(channelsNum)]);

            L(lChannelLoopBegin);
            cmp(rChannel, 0);
            jle(lChannelLoopEnd, T_NEAR);
        }

        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            if (isa == x64::avx512_core && tail)
                uni_kandd(kAuxMask, kTailMask, kGatherMask);
            else
                uni_kmovd(kAuxMask, kGatherMask);
        }

        if (!tail) {
            gatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask, useMask, zeroFill);
            uni_vmovups(ptr[rDstTmp], vAux);
        } else {
            if (isa == x64::avx512_core) {
                if (jcp.paddingMode != GridSamplePaddingMode::ZEROS) {
                    uni_kmovd(kAuxMask, kTailMask);
                }
                gatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask, tail, zeroFill);
                uni_vmovups(ptr[rDstTmp] | Xbyak::Opmask(kTailMask.getIdx()), vAux);
            } else {
                memMovDD(rDstTmp, rSrcTmp, Vmm(kAuxMask.getIdx()), vSrcShift, regWorkAmount, useMask, zeroFill);
            }
        }

        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);

        if (jcp.dynamicChannel) {
            dec(rChannel);
            jmp(lChannelLoopBegin, T_NEAR);
            L(lChannelLoopEnd);
        }
    }
}

template <>
void GridSampleKernel<x64::avx512_core>::bilinearInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    const auto& vDX = vWCoord;
    const auto& vDY = vHCoord;
    auto shift00    = getVmm();
    auto shift01    = getVmm();
    auto shift10    = getVmm();
    auto shift11    = getVmm();
    auto vAux       = getVmm();
    RegistersPool::Reg<Vmask> kMask00, kMask01, kMask10, kMask11;

    uni_vroundps(shift00, vWCoord, 0x1); // Round floor
    uni_vroundps(shift01, vHCoord, 0x1); // Round floor
    uni_vsubps(vDX, vWCoord, shift00);
    uni_vsubps(vDY, vHCoord, shift01);
    uni_vaddps(shift10, shift00, vOnesF);
    uni_vaddps(shift11, shift01, vOnesF);

    bool useMask = false, zeroFill = false;
    if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
        useMask = zeroFill = true;
        kMask00 = getMask();
        kMask01 = getMask();
        kMask10 = getMask();
        kMask11 = getMask();

        zerosPadding(kMask00, shift01, shift00); // (y; x)
        zerosPadding(kMask01, shift01, shift10); // (y; x + 1)
        zerosPadding(kMask11, shift11, shift10); // (y + 1; x + 1)
        zerosPadding(kMask10, shift11, shift00); // (y + 1; x)

        hwShiftPs2dq(shift00, shift01, shift00, vSrcWidthF);
        uni_vpaddd(shift01, shift00, vDataTypeSizeB);
        uni_vpaddd(shift10, shift00, vSrcWidthB);
        uni_vpaddd(shift11, shift10, vDataTypeSizeB);
    } else if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
        borderPadding(shift00, shift00, coord::w);
        borderPadding(shift01, shift01, coord::h);
        borderPadding(shift10, shift10, coord::w);
        borderPadding(shift11, shift11, coord::h);
    } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
        reflectionPadding(shift00, shift00, coord::w);
        reflectionPadding(shift01, shift01, coord::h);
        reflectionPadding(shift10, shift10, coord::w);
        reflectionPadding(shift11, shift11, coord::h);
    }
    if (jcp.paddingMode == GridSamplePaddingMode::BORDER || jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
        // W * y + x
        hwShiftPs2dq(vAux, shift11, shift00, vSrcWidthF);
        hwShiftPs2dq(shift00, shift01, shift00, vSrcWidthF);
        hwShiftPs2dq(shift01, shift01, shift10, vSrcWidthF);
        hwShiftPs2dq(shift11, shift11, shift10, vSrcWidthF);
        uni_vmovups(shift10, vAux);
    }

    auto kAuxMask = getMask();
    auto vQ0 = getVmm();
    auto vQ1 = getVmm();

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    RegistersPool::Reg<Xbyak::Reg64> rChannel;
    auto rSrcTmp  = getReg64();
    auto rDstTmp  = getReg64();
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);

    for (uint64_t ch = 0; ch < jcp.cannelNum; ch++) {
        if (jcp.dynamicChannel) {
            rChannel = getReg64();
            mov(rChannel, 0);

            L(lChannelLoopBegin);
            cmp(rChannel, regChannelNum);
            jge(lChannelLoopEnd, T_NEAR);
        }

        // (y; x)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask00);
        }
        gatherdd(vQ0, rSrcTmp, shift00, kAuxMask, useMask, zeroFill); // v00 -> vQ0
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vQ0, vQ0);
        }
        uni_vfmsub213ps(vQ0, vDX, vQ0); // q0 = -(v00 - dx * v00)

        // (y; x + 1)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask01);
        }
        gatherdd(vAux, rSrcTmp, shift01, kAuxMask, useMask, zeroFill);
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vAux, vAux);
        }
        uni_vfmsub231ps(vQ0, vAux, vDX); // q0 = -q0 + dx * v01

        // (y + 1; x + 1)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask11);
        }
        gatherdd(vAux, rSrcTmp, shift11, kAuxMask, useMask, zeroFill);
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vAux, vAux);
        }

        // (y + 1; x)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask10);
        }
        gatherdd(vQ1, rSrcTmp, shift10, kAuxMask, useMask, zeroFill);
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vQ1, vQ1);
        }

        uni_vfmsub213ps(vQ1, vDX, vQ1);  // q1 = -(v10 - dx * v10)
        uni_vfmsub231ps(vQ1, vAux, vDX); // q1 = -q1 + dx * v11
        // Res = q0 + dy * (q1 - q0)
        uni_vsubps(vQ1, vQ1, vQ0);
        uni_vfmadd132ps(vQ1, vQ0, vDY);

        if (jcp.inDataPrc == ov::element::i32) {
            uni_vroundps(vQ1, vQ1, 0x3); // Truncation
            uni_vcvtps2dq(vQ1, vQ1);
        }

        if (!tail) {
            uni_vmovups(ptr[rDstTmp], vQ1);
        } else {
            uni_vmovups(ptr[rDstTmp] | kTailMask, vQ1);
        }
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);

        if (jcp.dynamicChannel) {
            inc(rChannel);
            jmp(lChannelLoopBegin, T_NEAR);
            L(lChannelLoopEnd);
        }
    }
}

template <x64::cpu_isa_t isa> // Works for AVX2, AVX, SSE41
void GridSampleKernel<isa>::bilinearInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    auto vWRound = getVmm();
    auto vHRound = getVmm();
    auto& vDX    = vWCoord;
    auto& vDY    = vHCoord;
    auto vAux    = getVmm();
    Vmm shift00, shift01, shift10, shift11;
    RegistersPool::Reg<Vmm> shift10Holder, shift11Holder;
    // For ZEROS padding only.
    RegistersPool::Reg<Vmm> vMask00, vMask01, vMask10, vMask11;

    uni_vroundps(vWRound, vWCoord, 0x1); // Round floor
    uni_vroundps(vHRound, vHCoord, 0x1); // Round floor
    uni_vsubps(vDX, vDX, vWRound);
    uni_vsubps(vDY, vDY, vHRound);

    if (jcp.paddingMode != GridSamplePaddingMode::ZEROS) {
        shift00 = vWRound;
        shift01 = vHRound;
        shift10Holder = getVmm();
        shift10 = shift10Holder;
        shift11Holder = getVmm();
        shift11 = shift11Holder;

        uni_vaddps(shift10, vWRound, vOnesF);
        uni_vaddps(shift11, vHRound, vOnesF);
    }

    bool useMask = false, zeroFill = false;
    if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
        useMask = zeroFill = true;
        {
            auto rAux = getReg64();
            static const float onesArr[8] = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
            if (isa ==x64::sse41) {
                static const float *onesPtr = onesArr + (reinterpret_cast<int64_t>(onesArr) % 16) / sizeof(float);
                mov(rAux, reinterpret_cast<uintptr_t>(onesPtr));
            } else {
                mov(rAux, reinterpret_cast<uintptr_t>(onesArr));
            }
            uni_vmovups(vAux, ptr[rAux]);
        }
        shift00 = vWRound;
        shift10 = vHRound;
        vMask00 = getVmm();
        vMask01 = getVmm();
        vMask10 = getVmm();
        vMask11 = getVmm();

        uni_vaddps(vMask00, vWRound, vAux);
        uni_vaddps(vAux, vAux, vHRound);

        zerosPadding(vMask01, vHRound, vMask00); // (y; x + 1)
        zerosPadding(vMask10, vAux, vWRound);    // (y + 1; x)
        zerosPadding(vMask11, vAux, vMask00);    // (y + 1; x + 1)
        zerosPadding(vMask00, vHRound, vWRound); // (y; x)

        hwShiftPs2dq(shift00, vHRound, vWRound, vSrcWidthF);
    } else if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
        borderPadding(vWRound, vWRound, coord::w);
        borderPadding(vHRound, vHRound, coord::h);
        borderPadding(shift10, shift10, coord::w);
        borderPadding(shift11, shift11, coord::h);
    } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
        reflectionPadding(vWRound, vWRound, coord::w);
        reflectionPadding(vHRound, vHRound, coord::h);
        reflectionPadding(shift10, shift10, coord::w);
        reflectionPadding(shift11, shift11, coord::h);
    }
    if (one_of(jcp.paddingMode, GridSamplePaddingMode::BORDER, GridSamplePaddingMode::REFLECTION)) {
        // W * y + x
        hwShiftPs2dq(vAux, shift11, vWRound, vSrcWidthF);
        hwShiftPs2dq(vWRound, vHRound, vWRound, vSrcWidthF);
        hwShiftPs2dq(vHRound, vHRound, shift10, vSrcWidthF);
        hwShiftPs2dq(shift11, shift11, shift10, vSrcWidthF);
        uni_vmovups(shift10, vAux);
    }

    auto vGatherMask = getVmm();
    auto vQ0         = getVmm();
    auto vQ1         = getVmm();

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    RegistersPool::Reg<Xbyak::Reg64> rChannel;
    auto rSrcTmp   = getReg64();
    auto rDstTmp   = getReg64();
    auto rTypeSize = getReg64();
    mov(rSrcTmp,   regSrc);
    mov(rDstTmp,   regDst);
    mov(rTypeSize, ptr[regParams + GET_OFF(dataTypeSize)]);

    for (uint64_t ch = 0; ch < jcp.cannelNum; ch++) {
        if (jcp.dynamicChannel) {
            rChannel = getReg64();
            mov(rChannel, ptr[regParams + GET_OFF(channelsNum)]);

            L(lChannelLoopBegin);
            cmp(rChannel, 0);
            jle(lChannelLoopEnd, T_NEAR);
        }

        // (y; x)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS && isa == x64::avx2) {
            uni_vmovups(vGatherMask, vMask00);
        }
        gatherdd(vQ0, rSrcTmp, shift00, (isa == x64::avx2 || !vMask00.isInitialized()) ? vGatherMask : vMask00, useMask, zeroFill); // v00 -> vQ0
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vQ0, vQ0);
        }
        if (isa == x64::avx2) {
            uni_vfmsub213ps(vQ0, vDX, vQ0); // q0 = -(v00 - dx * v00)
        } else {
            uni_vmulps(vGatherMask, vQ0, vDX);
            uni_vsubps(vQ0, vQ0, vGatherMask);
        }

        // (y; x + 1)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            uni_vpaddd(shift10, shift00, ptr[rTypeSize]);
            if (isa == x64::avx2)
                uni_vmovups(vGatherMask, vMask01);
        }
        gatherdd(vAux, rSrcTmp, jcp.paddingMode != GridSamplePaddingMode::ZEROS ? shift01 : shift10,
                 (isa == x64::avx2 || !vMask01.isInitialized()) ? vGatherMask : vMask01, useMask, zeroFill);
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vAux, vAux);
        }
        if (isa == x64::avx2) {
            uni_vfmsub231ps(vQ0, vAux, vDX); // q0 = -q0 + dx * v01
        } else {
            uni_vmulps(vAux, vAux, vDX);
            uni_vaddps(vQ0, vQ0, vAux);
        }

        // (y + 1; x + 1)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            {
                auto rSrcWidth = getReg64();
                mov(rSrcWidth, ptr[regParams + GET_OFF(srcWidthB)]);
                uni_vpaddd(shift10, shift10, ptr[rSrcWidth]);
            }
            if (isa == x64::avx2)
                uni_vmovups(vGatherMask, vMask11);
        }
        gatherdd(vAux, rSrcTmp, jcp.paddingMode != GridSamplePaddingMode::ZEROS ? shift11 : shift10,
                 (isa == x64::avx2 || !vMask11.isInitialized()) ? vGatherMask : vMask11, useMask, zeroFill);
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vAux, vAux);
        }

        // (y + 1; x)
        if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
            uni_vpsubd(shift10, shift10, ptr[rTypeSize]);
            if (isa == x64::avx2)
                uni_vmovups(vGatherMask, vMask10);
        }
        gatherdd(vQ1, rSrcTmp, shift10, (isa == x64::avx2 || !vMask10.isInitialized()) ? vGatherMask : vMask10, useMask, zeroFill);
        if (jcp.inDataPrc == ov::element::i32) {
            uni_vcvtdq2ps(vQ1, vQ1);
        }

        // q1 = -(v10 - dx * v10)
        if (isa == x64::avx2) {
            uni_vfmsub213ps(vQ1, vDX, vQ1);
        } else {
            uni_vmulps(vGatherMask, vQ1, vDX);
            if (isa == x64::avx) {
                uni_vsubps(vQ1, vGatherMask, vQ1);
            } else {
                uni_vsubps(vGatherMask, vGatherMask, vQ1);
                uni_vmovups(vQ1, vGatherMask);
            }
        }
        uni_vfmsub231ps(vQ1, vAux, vDX); // q1 = -q1 + dx * v11
        // Res = q0 + dy * (q1 - q0)
        uni_vsubps(vQ1, vQ1, vQ0);
        uni_vfmadd132ps(vQ1, vQ0, vDY);

        if (jcp.inDataPrc == ov::element::i32) {
            uni_vroundps(vQ1, vQ1, 0x3); // Truncation
            uni_vcvtps2dq(vQ1, vQ1);
        }

        if (!tail) {
            uni_vmovups(ptr[rDstTmp], vQ1);
        } else {
            store(ptr[rDstTmp], vQ1, regWorkAmount, dataTypeSize);
        }

        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);

        if (jcp.dynamicChannel) {
            dec(rChannel);
            jmp(lChannelLoopBegin, T_NEAR);
            L(lChannelLoopEnd);
        }
    }
}

template <>
void GridSampleKernel<x64::avx512_core>::bicubicInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    auto vHTop      = getVmm();
    auto vWLeft     = getVmm();
    auto vDX        = getVmm();
    auto vDY        = getVmm();
    auto vXDotProd  = getVmm();
    auto& vYDotProd = vDX;
    auto vSrcShift0 = getVmm();
    auto vSrcShift  = getVmm();
    auto vAux       = getVmm();
    auto kAuxMask   = getMask();
    RegistersPool::Reg<Vmask> kMaskH;
    std::vector<RegistersPool::Reg<Vmask>> wMasks;

    uni_vroundps(vHTop, vHCoord, 0x1);  // Round floor
    uni_vroundps(vWLeft, vWCoord, 0x1); // Round floor
    uni_vsubps(vDY, vHCoord, vHTop);
    uni_vsubps(vDX, vWCoord, vWLeft);
    uni_vsubps(vHTop, vHTop, vOnesF);
    uni_vsubps(vWLeft, vWLeft, vOnesF);

    RegistersPool::Reg<Vmm> vCX[4] = {getVmm(), getVmm(), getVmm(), getVmm() };
    for (int i = 0; i < 4; i++) {
        bicubicCoefficients(vCX[i], vDX, i);
    }

    bool useMask = false, zeroFill = false;
    if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
        useMask = zeroFill = true;
        wMasks.resize(4);
        for (auto& mask : wMasks) {
            mask = getMask();
        }
        zerosPaddingW(wMasks[0], vWLeft);
        uni_vaddps(vWCoord, vWLeft, vOnesF);
        zerosPaddingW(wMasks[1], vWCoord);
        uni_vaddps(vWCoord, vWCoord, vOnesF);
        zerosPaddingW(wMasks[2], vWCoord);
        uni_vaddps(vWCoord, vWCoord, vOnesF);
        zerosPaddingW(wMasks[3], vWCoord);
        kMaskH = getMask();
    }

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    RegistersPool::Reg<Xbyak::Reg64> rChannel;
    auto rSrcTmp  = getReg64();
    auto rDstTmp  = getReg64();
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);

    for (size_t ch = 0; ch < jcp.cannelNum; ch++) {
        if (jcp.dynamicChannel) {
            rChannel = getReg64();
            mov(rChannel, 0);

            L(lChannelLoopBegin);
            cmp(rChannel, regChannelNum);
            jge(lChannelLoopEnd, T_NEAR);
        }

        uni_vmovups(vHCoord, vHTop);
        uni_vpxor(vYDotProd, vYDotProd, vYDotProd);
        for (int h = 0; h < 4; h++) {
            // (y - 1 + h; x - 1)
            if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
                Xbyak::Opmask maskH = kMaskH;
                vcmpps(kMaskH, vHCoord, vSrcHeightF, CMP_LT_PS);
                vcmpps(maskH | maskH, vZeros, vHCoord, CMP_LE_PS);
                kandw(kAuxMask, kMaskH, wMasks[0]);
                uni_vmulps(vSrcShift0, vHCoord, vSrcWidthF);
                uni_vmovups(vWCoord, vWLeft);
                uni_vaddps(vSrcShift, vSrcShift0, vWCoord);
            } else if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
                borderPadding(vSrcShift0, vHCoord, coord::h);
                uni_vmulps(vSrcShift0, vSrcShift0, vSrcWidthF);
                uni_vmovups(vWCoord, vWLeft);
                borderPadding(vSrcShift, vWCoord, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
            } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
                reflectionPadding(vSrcShift0, vHCoord, coord::h);
                uni_vmulps(vSrcShift0, vSrcShift0, vSrcWidthF);
                uni_vmovups(vWCoord, vWLeft);
                reflectionPadding(vSrcShift, vWCoord, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
            }
            uni_vcvtps2dq(vSrcShift, vSrcShift);
            if (dataTypeSize > 1)
                uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            gatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask, useMask, zeroFill);
            if (jcp.inDataPrc == ov::element::i32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vmulps(vXDotProd, vAux, vCX[0]);

            // (y - 1 + h; x)
            // (y - 1 + h; x + 1)
            // (y - 1 + h; x + 2)
            for (int w = 1; w < 4; w++) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
                    uni_vaddps(vSrcShift, vSrcShift0, vWCoord);
                    kandw(kAuxMask, kMaskH, wMasks[w]);
                } else if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
                    borderPadding(vSrcShift, vWCoord, coord::w);
                    uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
                    reflectionPadding(vSrcShift, vWCoord, coord::w);
                    uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                }
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
                gatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask, useMask, zeroFill);
                if (jcp.inDataPrc == ov::element::i32) {
                    uni_vcvtdq2ps(vAux, vAux);
                }
                uni_vfmadd231ps(vXDotProd, vAux, vCX[w]);
            }

            if (h != 3) {
                uni_vaddps(vHCoord, vHCoord, vOnesF);
            }

            bicubicCoefficients(vAux, vDY, h);
            uni_vfmadd231ps(vYDotProd, vXDotProd, vAux);
        }

        if (jcp.inDataPrc == ov::element::i32) {
            uni_vroundps(vYDotProd, vYDotProd, 0x3); // Truncation
            uni_vcvtps2dq(vYDotProd, vYDotProd);
        }

        if (!tail) {
            uni_vmovups(ptr[rDstTmp], vYDotProd);
        } else {
            uni_vmovups(ptr[rDstTmp] | kTailMask, vYDotProd);
        }
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);

        if (jcp.dynamicChannel) {
            inc(rChannel);
            jmp(lChannelLoopBegin, T_NEAR);
            L(lChannelLoopEnd);
        }
    }
}

template <x64::cpu_isa_t isa> // Works for AVX2, AVX, SSE41
void GridSampleKernel<isa>::bicubicInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    auto vHTop  = getVmm();
    auto vWLeft = getVmm();
    auto vDX    = getVmm();
    auto vDY    = getVmm();

    uni_vroundps(vHTop,  vHCoord, 0x1); // Round floor
    uni_vroundps(vWLeft, vWCoord, 0x1); // Round floor
    uni_vsubps(vDY, vHCoord, vHTop);
    uni_vsubps(vDX, vWCoord, vWLeft);
    uni_vsubps(vHTop, vHTop, vOnesF);
    uni_vsubps(vWLeft, vWLeft, vOnesF);

    auto rBuff = getReg64();
    mov(rBuff, ptr[regParams + GET_OFF(buffer)]);

    bool useMask = false, zeroFill = false;

    if (jcp.paddingMode == GridSamplePaddingMode::BORDER) {
        auto rAux = getReg64();

        if (!vSrcWidthSub1F.isInitialized()) {
            vSrcWidthSub1F = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(srcWidthSub1F)]);
            uni_vmovups(vSrcWidthSub1F, ptr[rAux]);
        }

        auto vW0 = getVmm(), vW1 = getVmm();
        Vmm vW[4] = { vW0, vW1, vHCoord, vWCoord };
        for (int w = 0; w < 4; w++) {
            borderPadding(vW[w], vWLeft, coord::w);
            if (w < 3) {
                uni_vaddps(vWLeft, vWLeft, vOnesF);
            }
        }
        vWLeft.release();
        vSrcWidthSub1F.release();

        if (!vSrcHeightSub1F.isInitialized()) {
            vSrcHeightSub1F = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(srcHeightSub1F)]);
            uni_vmovups(vSrcHeightSub1F, ptr[rAux]);
        }
        auto vH  = getVmm();

        size_t bufShift = 0lu;
        for (int h = 0; h < 4; h++) {
            borderPadding(vH, vHTop, coord::h);
            uni_vmulps(vH, vH, vSrcWidthF);
            auto vShift = getVmm();
            for (int w = 0; w < 4; w++) {
                uni_vaddps(vShift, vH, vW[w]);
                dataTypeShiftPs2Dq(vShift, vShift);
                uni_vmovups(ptr[rBuff + bufShift], vShift);
                bufShift += vlen;
            }
            if (h < 3) {
                uni_vaddps(vHTop, vHTop, vOnesF);
            }
        }
        vHTop.release();
        vSrcHeightSub1F.release();
    } else if (jcp.paddingMode == GridSamplePaddingMode::REFLECTION) {
        auto rAux = getReg64();
        if (!jcp.alignCorners && !vSrcWidthMul2F.isInitialized()) {
            vSrcWidthMul2F = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2F)]);
            uni_vmovups(vSrcWidthMul2F, ptr[rAux]);
        }
        if (!vSrcWidthMul2Sub1F.isInitialized()) {
            vSrcWidthMul2Sub1F = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
            uni_vmovups(vSrcWidthMul2Sub1F, ptr[rAux]);
        }

        auto vW0 = getVmm(), vW1 = getVmm();
        Vmm vW[4] = { vW0, vW1, vHCoord, vWCoord };
        for (int w = 0; w < 4; w++) {
            reflectionPadding(vW[w], vWLeft, coord::w);
            if (w < 3) {
                uni_vaddps(vWLeft, vWLeft, vOnesF);
            }
        }
        vWLeft.release();
        vSrcWidthMul2F.release();
        vSrcWidthMul2Sub1F.release();

        if (!jcp.alignCorners && !vSrcHeightMul2F.isInitialized()) {
            vSrcHeightMul2F = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2F)]);
            uni_vmovups(vSrcHeightMul2F, ptr[rAux]);
        }
        if (!vSrcHeightMul2Sub1F.isInitialized()) {
            vSrcHeightMul2Sub1F = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
            uni_vmovups(vSrcHeightMul2Sub1F, ptr[rAux]);
        }
        auto vH  = getVmm();

        size_t bufShift = 0lu;
        for (int h = 0; h < 4; h++) {
            reflectionPadding(vH, vHTop, coord::h);
            uni_vmulps(vH, vH, vSrcWidthF);
            auto vShift = getVmm();
            for (int w = 0; w < 4; w++) {
                uni_vaddps(vShift, vH, vW[w]);
                dataTypeShiftPs2Dq(vShift, vShift);
                uni_vmovups(ptr[rBuff + bufShift], vShift);
                bufShift += vlen;
            }
            if (h < 3) {
                uni_vaddps(vHTop, vHTop, vOnesF);
            }
        }
        vHTop.release();
        vSrcHeightMul2F.release();
        vSrcHeightMul2Sub1F.release();
    } else if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
        useMask = zeroFill = true;

        RegistersPool::Reg<Vmm> vWMask[4] = { getVmm(), getVmm(), getVmm(), getVmm() };
        for (int w = 0; w < 4; w++) {
            if (w == 0) {
                zerosPaddingW(vWMask[w], vWLeft);
                uni_vaddps(vWCoord, vWLeft, vOnesF);
            } else {
                zerosPaddingW(vWMask[w], vWCoord);
                if (w < 3) {
                    uni_vaddps(vWCoord, vWCoord, vOnesF);
                }
            }
        }

        size_t bufShift = 0lu;
        auto vShift = vWCoord, vMaskH = vHCoord;
        if (!vDataTypeSizeB.isInitialized()) {
            auto rAux = getReg64();
            vDataTypeSizeB = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(dataTypeSize)]);
            uni_vmovups(vDataTypeSizeB, ptr[rAux]);
        }

        for (int h = 0; h < 4; h++) {
            if (isa == x64::avx2) {
                uni_vmovups(vShift, vHTop);
                uni_vfmadd132ps(vShift, vWLeft, vSrcWidthF);
            } else {
                uni_vmulps(vShift, vHTop, vSrcWidthF);
                uni_vaddps(vShift, vShift, vWLeft);
            }
            dataTypeShiftPs2Dq(vShift, vShift);
            for (int w = 0; w < 4; w++) {
                uni_vmovups(ptr[rBuff + bufShift], vShift);
                if (w < 3) {
                    uni_vpaddd(vShift, vShift, vDataTypeSizeB);
                }

                zerosPaddingH(vMaskH, vHTop, vWMask[w]);
                uni_vmovups(ptr[rBuff + bufShift + 16 * vlen], vMaskH);
                bufShift += vlen;
            }
            if (h < 3) {
                uni_vaddps(vHTop, vHTop, vOnesF);
            }
        }
        vHTop.release();
        vWLeft.release();
        vDataTypeSizeB.release();
    }

    RegistersPool::Reg<Vmm> vCX[4] = { getVmm(), getVmm(), getVmm(), getVmm() };
    for (int w = 0; w < 4; w++) {
        bicubicCoefficients(vCX[w], vDX, w);
    }
    auto vCY0 = getVmm(), vCY1 = getVmm();
    Vmm vCY[4] = { vCY0, vCY1, vHCoord, vWCoord };
    for (int h = 0; h < 4; h++) {
        bicubicCoefficients(vCY[h], vDY, h);
    }

    const auto& vXDotProd = vDX;
    const auto& vYDotProd = vDY;
    auto vSrcShift   = getVmm();
    auto kGatherMask = getVmm();
    auto vAux        = getVmm();

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    RegistersPool::Reg<Xbyak::Reg64> rChannel;
    auto rSrcTmp = getReg64();
    auto rDstTmp = getReg64();
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);

    for (uint64_t ch = 0; ch < jcp.cannelNum; ch++) {
        if (jcp.dynamicChannel) {
            rChannel = getReg64();
            mov(rChannel, ptr[regParams + GET_OFF(channelsNum)]);

            L(lChannelLoopBegin);
            cmp(rChannel, 0);
            jle(lChannelLoopEnd, T_NEAR);
        }

        uni_vpxor(vYDotProd, vYDotProd, vYDotProd);
        for (int h = 0; h < 4; h++) {
            size_t bufShift = h * 4 * vlen;
            // (y - 1 + h; x - 1)
            if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
                uni_vmovups(kGatherMask, ptr[rBuff + bufShift + 16 * vlen]);
            }
            uni_vmovups(vSrcShift, ptr[rBuff + bufShift]);
            bufShift += vlen;

            gatherdd(vAux, rSrcTmp, vSrcShift, kGatherMask, useMask, zeroFill);
            if (jcp.inDataPrc == ov::element::i32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vmulps(vXDotProd, vAux, vCX[0]);

            // (y - 1 + h; x)
            // (y - 1 + h; x + 1)
            // (y - 1 + h; x + 2)
            for (int w = 1; w < 4; w++) {
                if (jcp.paddingMode == GridSamplePaddingMode::ZEROS) {
                    uni_vmovups(kGatherMask, ptr[rBuff + bufShift + 16 * vlen]);
                }
                uni_vmovups(vSrcShift, ptr[rBuff + bufShift]);
                bufShift += vlen;

                gatherdd(vAux, rSrcTmp, vSrcShift, kGatherMask, useMask, zeroFill);
                if (jcp.inDataPrc == ov::element::i32) {
                    uni_vcvtdq2ps(vAux, vAux);
                }
                uni_vfmadd231ps(vXDotProd, vAux, vCX[w]);
            }
            uni_vfmadd231ps(vYDotProd, vXDotProd, vCY[h]);
        }

        if (jcp.inDataPrc == ov::element::i32) {
            uni_vroundps(vYDotProd, vYDotProd, 0x3); // Truncation
            uni_vcvtps2dq(vYDotProd, vYDotProd);
        }

        if (!tail) {
            uni_vmovups(ptr[rDstTmp], vYDotProd);
        } else {
            store(ptr[rDstTmp], vYDotProd, regWorkAmount, dataTypeSize);
        }
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);

        if (jcp.dynamicChannel) {
            dec(rChannel);
            jmp(lChannelLoopBegin, T_NEAR);
            L(lChannelLoopEnd);
        }
    }
}

template <x64::cpu_isa_t isa>
void GridSampleKernel<isa>::dataTypeShiftPs2Dq(const Vmm& vDst, const Vmm& vSrc) {
    if (dataTypeSize == 1)
        return;

    if (isa == x64::avx) { // vpslld works just with XMM for AVX, so use vmulps for YMM
        auto rAux = getReg64();
        static const float val = dataTypeSize;
        static const float dataTypeSizeArr[8] = {val, val, val, val, val, val, val, val};
        mov(rAux, reinterpret_cast<uintptr_t>(dataTypeSizeArr));
        uni_vmulps(vDst, vSrc, ptr[rAux]);
        uni_vcvtps2dq(vDst, vDst);
    } else {
        uni_vcvtps2dq(vDst, vSrc);
        if (dataTypeSize > 1)
            uni_vpslld(vDst, vDst, dataTypeShift); // multiply by source data type size.
    }
}

template <x64::cpu_isa_t isa>
void GridSampleKernel<isa>::hwShiftPs2dq(const Vmm& vDst, const Vmm& vHCoord, const Vmm& vWCoord, const Vmm& vWidth) {
    if (vDst.getIdx() == vWCoord.getIdx()) {
        if (one_of(isa, x64::avx512_core, x64::avx2)) {
            uni_vfmadd231ps(vDst, vHCoord, vWidth);
        } else {
            auto vTmp = getVmm();
            uni_vmulps(vTmp, vHCoord, vWidth);
            uni_vaddps(vDst, vWCoord, vTmp);
        }
    } else if (vDst.getIdx() == vHCoord.getIdx()) {
        uni_vfmadd132ps(vDst, vWCoord, vWidth);
    } else if (vDst.getIdx() == vWidth.getIdx()) {
        uni_vfmadd132ps(vDst, vWCoord, vHCoord);
    } else {
        if (one_of(isa, x64::avx2, x64::avx512_core)) {
            uni_vmovups(vDst, vWCoord);
            uni_vfmadd231ps(vDst, vHCoord, vWidth);
        } else {
            uni_vmulps(vDst, vHCoord, vWidth);
            uni_vaddps(vDst, vDst, vWCoord);
        }
    }

    if (isa == x64::avx) { // vpslld works just with XMM for AVX, so use vmulps for YMM
        if (dataTypeSize > 1) {
            auto rAux = getReg64();
            const float val = dataTypeSize;
            static const float dataTypeSizeArr[8] = {val, val, val, val, val, val, val, val};
            mov(rAux, reinterpret_cast<uintptr_t>(dataTypeSizeArr));
            uni_vmulps(vDst, vDst, ptr[rAux]);
        }
        uni_vcvtps2dq(vDst, vDst);
    } else {
        uni_vcvtps2dq(vDst, vDst);
        if (dataTypeSize > 1)
            uni_vpslld(vDst, vDst, dataTypeShift); // multiply by source data type size.
    }
}

template class GridSampleKernel<x64::avx512_core>;
template class GridSampleKernel<x64::avx2>;
template class GridSampleKernel<x64::sse41>;

}   // namespace kernel
}   // namespace intel_cpu
}   // namespace ov
