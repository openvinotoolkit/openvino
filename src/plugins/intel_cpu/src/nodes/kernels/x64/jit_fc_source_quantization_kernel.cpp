// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_fc_source_quantization_kernel.hpp"

#include "openvino/core/except.hpp"

namespace ov::intel_cpu {

template <cpu_isa_t isa>
JitFCSourceQuantizationKernel<isa>::JitFCSourceQuantizationKernel(const FCSourceQuantizationKernelCompileParams& params)
    : jit_generator_t(jit_name()),
      m_groupSize(params.groupSize),
      m_blockSize(cpu_isa_traits_t<isa>::vlen / sizeof(float)) {
    OPENVINO_ASSERT(m_groupSize != 0 && m_groupSize % m_blockSize == 0,
                    "JIT source quantization group size must be a multiple of the vector width");
    create_kernel();
    m_kernel = reinterpret_cast<decltype(m_kernel)>(jit_ker());
}

template <cpu_isa_t isa>
void JitFCSourceQuantizationKernel<isa>::operator()(const FCSourceQuantizationKernelRuntimeParams* args) const {
    OPENVINO_ASSERT(m_kernel != nullptr, "JIT source quantization kernel is not initialized");
    m_kernel(args);
}

template <cpu_isa_t isa>
size_t JitFCSourceQuantizationKernel<isa>::blockSize() const {
    return m_blockSize;
}

template <cpu_isa_t isa>
void JitFCSourceQuantizationKernel<isa>::reduceMax(Vmm value, Vmm temp) {
    if (isa == avx512_core) {
        const Xbyak::Zmm zmmValue(value.getIdx());
        const Xbyak::Zmm zmmTemp(temp.getIdx());
        vshuff32x4(zmmTemp, zmmValue, zmmValue, 0x4E);
        uni_vmaxps(zmmValue, zmmValue, zmmTemp);
        vshuff32x4(zmmTemp, zmmValue, zmmValue, 0xB1);
        uni_vmaxps(zmmValue, zmmValue, zmmTemp);
    } else {
        const Xbyak::Ymm ymmValue(value.getIdx());
        const Xbyak::Ymm ymmTemp(temp.getIdx());
        vperm2i128(ymmTemp, ymmValue, ymmValue, 0x01);
        uni_vmaxps(ymmValue, ymmValue, ymmTemp);
    }

    uni_vshufps(temp, value, value, 0x4E);
    uni_vmaxps(value, value, temp);
    uni_vshufps(temp, value, value, 0xB1);
    uni_vmaxps(value, value, temp);
}

template <cpu_isa_t isa>
void JitFCSourceQuantizationKernel<isa>::storeQuantized(size_t offset) {
    if (isa == avx512_core) {
        vpmovsdb(ptr[regDst + offset], vmmValues());
        return;
    }

    uni_vpackssdw(vmmValues(), vmmValues(), vmmValues());
    vpermq(Xbyak::Ymm(vmmValues().getIdx()), Xbyak::Ymm(vmmValues().getIdx()), 0x08);
    uni_vpacksswb(vmmValues(), vmmValues(), vmmValues());
    vmovq(ptr[regDst + offset], Xbyak::Xmm(vmmValues().getIdx()));
}

template <cpu_isa_t isa>
void JitFCSourceQuantizationKernel<isa>::zeroDstBlock(size_t offset) {
    mov(qword[regDst + offset], 0);
    if (m_blockSize == 16) {
        mov(qword[regDst + offset + 8], 0);
    }
}

template <cpu_isa_t isa>
void JitFCSourceQuantizationKernel<isa>::generate() {
    preamble();

    static const float negativeZero[16] = {
        -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f,
        -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f};
    static const float positiveOne[16] = {
        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    static const float int8Max[16] = {
        127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f,
        127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f};

    mov(regSrc, ptr[param1 + offSrc()]);
    mov(regDst, ptr[param1 + offDst()]);
    mov(regScale, ptr[param1 + offScale()]);
    mov(regGroupCount, ptr[param1 + offGroupCount()]);

    mov(regTmp, reinterpret_cast<size_t>(negativeZero));
    uni_vmovups(vmmSignMask(), ptr[regTmp]);

    mov(regTmp, reinterpret_cast<size_t>(positiveOne));
    uni_vmovups(vmmOne(), ptr[regTmp]);

    mov(regTmp, reinterpret_cast<size_t>(int8Max));
    uni_vmovups(vmmInt8Max(), ptr[regTmp]);

    const size_t blocks = m_groupSize / m_blockSize;
    const size_t srcGroupStep = m_groupSize * sizeof(float);
    const size_t dstGroupStep = m_groupSize * sizeof(int8_t);

    Xbyak::Label groupLoop;
    Xbyak::Label zeroScale;
    Xbyak::Label nextGroup;
    Xbyak::Label done;

    L(groupLoop);
    cmp(regGroupCount, 0);
    je(done, T_NEAR);

    uni_vxorps(vmmMax(), vmmMax(), vmmMax());
    for (size_t block = 0; block < blocks; block++) {
        const size_t byteOffset = block * m_blockSize * sizeof(float);
        uni_vmovups(vmmValues(), ptr[regSrc + byteOffset]);
        vandnps(vmmValues(), vmmSignMask(), vmmValues());
        uni_vmaxps(vmmMax(), vmmMax(), vmmValues());
    }

    reduceMax(vmmMax(), vmmTmp());
    uni_vxorps(vmmZero(), vmmZero(), vmmZero());
    vucomiss(Xbyak::Xmm(vmmMax().getIdx()), Xbyak::Xmm(vmmZero().getIdx()));
    je(zeroScale, T_NEAR);

    uni_vbroadcastss(vmmScale(), Xbyak::Xmm(vmmMax().getIdx()));
    uni_vdivps(vmmScale(), vmmScale(), vmmInt8Max());
    uni_vmovss(ptr[regScale], Xbyak::Xmm(vmmScale().getIdx()));
    uni_vdivps(vmmQScale(), vmmOne(), vmmScale());

    for (size_t block = 0; block < blocks; block++) {
        const size_t srcOffset = block * m_blockSize * sizeof(float);
        const size_t dstOffset = block * m_blockSize * sizeof(int8_t);
        uni_vmovups(vmmValues(), ptr[regSrc + srcOffset]);
        uni_vmulps(vmmValues(), vmmValues(), vmmQScale());
        uni_vcvtps2dq(vmmValues(), vmmValues());
        storeQuantized(dstOffset);
    }
    jmp(nextGroup, T_NEAR);

    L(zeroScale);
    mov(dword[regScale], 0);
    for (size_t block = 0; block < blocks; block++) {
        zeroDstBlock(block * m_blockSize * sizeof(int8_t));
    }

    L(nextGroup);
    add(regSrc, srcGroupStep);
    add(regDst, dstGroupStep);
    add(regScale, sizeof(float));
    dec(regGroupCount);
    jmp(groupLoop, T_NEAR);

    L(done);
    postamble();
}

template class JitFCSourceQuantizationKernel<cpu_isa_t::avx2>;
template class JitFCSourceQuantizationKernel<cpu_isa_t::avx512_core>;

}  // namespace ov::intel_cpu
