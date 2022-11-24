// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_base.hpp"

using namespace ov::intel_cpu;
using namespace dnnl::impl::cpu::x64;


void jit_kernel_base::generate() {
    this->preamble();

    createRegistersPool();
    createStackAllocator();

    generate_impl();

    registersPool.reset();
    stackAllocator.reset();

    this->postamble();

    for (auto& record : emittersMap) {
        record.second->emit_data();
    }
}

void jit_kernel_base::emu_vgatherdps(const Xbyak::Xmm& vDst,
                                     const Xbyak::Reg64& rSrcPtr,
                                     const Xbyak::Xmm& vSrcShift,
                                     const Xbyak::Xmm& vReadMask,
                                     const bool useMask,
                                     const bool zeroFill) {
    std::vector<Xbyak::Xmm> not_available_xmm{vSrcShift, vDst, vReadMask};

    if (is_valid_isa(x64::avx512_core)) {
        RegistersPool::Reg<Xbyak::Opmask> avx512_mask{registersPool, 1};
        if (useMask) {
            const Xbyak::Zmm zmm_zero_val = registersPool->getInplaceFree<Xbyak::Zmm>(not_available_xmm);
            push(zmm_zero_val);
            uni_vxorps(zmm_zero_val, zmm_zero_val, zmm_zero_val);

            vpcmpud(avx512_mask, Xbyak::Zmm{vReadMask.getIdx()}, zmm_zero_val, VCMPPS_GT);
            pop(zmm_zero_val);
        }

        gatherdd(vDst, rSrcPtr, vSrcShift, avx512_mask, useMask, zeroFill);
    } else if (vDst.isYMM()) {
        Xbyak::Ymm yDst{vDst.getIdx()};
        Xbyak::Ymm yIndex{vSrcShift.getIdx()};
        Xbyak::Ymm yMask{vReadMask.getIdx()};

        gatherdd(yDst, rSrcPtr, yIndex, yMask, useMask, zeroFill);
    } else {
        gatherdd(vDst, rSrcPtr, vSrcShift, vReadMask, useMask, zeroFill);
    }
}

void jit_kernel_base::emu_vscatterdps(const Xbyak::Reg64& reg_addr,
                                    const Xbyak::Xmm& xmm_index,
                                    const Xbyak::Xmm& xmm_val,
                                    const Xbyak::Reg& reg_mask,
                                    const bool is_mask_seq /* = true*/) {
    const size_t kDataTypeSize = sizeof(float);
    Xbyak::Xmm xmm_mask{reg_mask.getIdx(), reg_mask.getKind(), static_cast<int>(reg_mask.getBit())};
    std::vector<Xbyak::Xmm> not_available_xmm{xmm_index, xmm_val, xmm_mask};
    if (is_valid_isa(x64::avx512_core)) {
        const Xbyak::Zmm zmm_zero_val = registersPool->getInplaceFree<Xbyak::Zmm>(not_available_xmm);
        push(zmm_zero_val);
        uni_vxorps(zmm_zero_val, zmm_zero_val, zmm_zero_val);
        RegistersPool::Reg<Xbyak::Opmask> avx512_mask{registersPool, 1};
        vpcmpud(avx512_mask, Xbyak::Zmm{reg_mask.getIdx()}, zmm_zero_val, VCMPPS_GT);
        pop(zmm_zero_val);
        vscatterdps(ptr[reg_addr + xmm_index], xmm_val | avx512_mask);
    } else {
        assert(reg_mask.isXMM() || reg_mask.isYMM());
        const size_t kXmmSimdWidth = x64::cpu_isa_traits<x64::sse41>::vlen / kDataTypeSize;
        const size_t kYmmSimdWidth = x64::cpu_isa_traits<x64::avx2>::vlen / kDataTypeSize;
        assert(xmm_val.getKind() == xmm_index.getKind());
        assert(xmm_index.getKind() == xmm_mask.getKind());

        std::vector<Xbyak::Reg> not_available_reg{reg_addr};
        const Xbyak::Reg64 idx = registersPool->getInplaceFree<Xbyak::Reg64>(not_available_reg);
        const Xbyak::Reg64 mask = registersPool->getInplaceFree<Xbyak::Reg64>(not_available_reg);
        const Xbyak::Reg64 val = registersPool->getInplaceFree<Xbyak::Reg64>(not_available_reg);
        const Xbyak::Xmm xmm_mask_temp = registersPool->getInplaceFree<Xbyak::Xmm>(not_available_xmm);
        const Xbyak::Xmm xmm_index_temp = registersPool->getInplaceFree<Xbyak::Xmm>(not_available_xmm);
        const Xbyak::Xmm xmm_val_temp = registersPool->getInplaceFree<Xbyak::Xmm>(not_available_xmm);

        push(idx);
        push(mask);
        push(val);
        if (is_valid_isa(x64::avx2)) {
            push(Xbyak::Ymm{xmm_mask_temp.getIdx()});
            push(Xbyak::Ymm{xmm_index_temp.getIdx()});
            push(Xbyak::Ymm{xmm_val_temp.getIdx()});
        }
        xor_(idx, idx);
        xor_(mask, mask);
        xor_(val, val);

        Xbyak::Label scatter_fast_end;
        auto store_xmm = [&](const Xbyak::Xmm& xmm_mask, const Xbyak::Xmm& xmm_index, const Xbyak::Xmm& xmm_val) {
            for (int i = 0; i < static_cast<int>(kXmmSimdWidth); i++) {
                Xbyak::Label scatter_end;
                uni_vpextrd(mask.cvt32(), xmm_mask, i);
                cmp(mask.cvt32(), 0xFFFFFFFF);
                if (is_mask_seq) {
                    jne(scatter_fast_end, T_NEAR);
                } else {
                    jne(scatter_end, T_NEAR);
                }
                uni_vpextrd(idx.cvt32(), xmm_index, i);
                Xbyak::Address addr = ptr[reg_addr + idx];
                uni_vpextrd(val.cvt32(), xmm_val, i);
                mov(addr, val.cvt32());
                if (!is_mask_seq) {
                    L(scatter_end);
                }
            }
        };

        if (is_valid_isa(x64::avx2)) {
            for (int i = 0; i < static_cast<int>(kYmmSimdWidth / kXmmSimdWidth); i++) {
                vextracti128(xmm_mask_temp, Xbyak::Ymm{xmm_mask.getIdx()}, i);
                vextracti128(xmm_index_temp, Xbyak::Ymm{xmm_index.getIdx()}, i);
                vextracti128(xmm_val_temp, Xbyak::Ymm{xmm_val.getIdx()}, i);
                store_xmm(xmm_mask_temp, xmm_index_temp, xmm_val_temp);
            }
        } else {
            store_xmm(xmm_mask, xmm_index, xmm_val);
        }
        L(scatter_fast_end);

        if (is_valid_isa(x64::avx2)) {
            pop(Xbyak::Ymm{xmm_val_temp.getIdx()});
            pop(Xbyak::Ymm{xmm_index_temp.getIdx()});
            pop(Xbyak::Ymm{xmm_mask_temp.getIdx()});
        }
        pop(val);
        pop(mask);
        pop(idx);
    }
}

void jit_kernel_base::gatherdd(const Xbyak::Xmm& vDst,
                               const Xbyak::Reg64& rSrcPtr,
                               const Xbyak::Xmm& vSrcShift,
                               const Xbyak::Opmask& kReadMask,
                               const bool useMask,
                               const bool zeroFill) {
    if (!is_valid_isa(x64::avx512_core)) {
        IE_THROW() << "The vpgatherdd instruction with Opmask must be used when AVX-512 is available";
    }
    if (kReadMask.getIdx() == 0) {
        IE_THROW() << "The vpgatherdd instruction cannot use the register k0 as mask.";
    }
    if (!useMask)
        kxnord(kReadMask, kReadMask, kReadMask);
    if (zeroFill)
        uni_vpxor(vDst, vDst, vDst);

    vpgatherdd(vDst | kReadMask, ptr[rSrcPtr + vSrcShift]);
}

void jit_kernel_base::gatherdd(const Xbyak::Xmm&   vDst,
                             const Xbyak::Reg64& rSrcPtr,
                             const Xbyak::Xmm&   vSrcShift,
                             const Xbyak::Xmm&   vReadMask,
                             const bool useMask,
                             const bool zeroFill) {
    if (vDst.getIdx() == vSrcShift.getIdx() || vDst.getIdx() == vReadMask.getIdx() || vSrcShift.getIdx() == vReadMask.getIdx()) {
        IE_THROW() << "Any pair of the index, mask, or destination registers cannot be the same.";
    }
    if (zeroFill)
        pxor(vDst, vDst); // Don't use vpxor. It zeros the rest of the YMM register.

    if (is_valid_isa(x64::avx2)) {
        if (!useMask)
            uni_vpcmpeqd(vReadMask, vReadMask, vReadMask);

        vpgatherdd(vDst, ptr[rSrcPtr + vSrcShift], vReadMask);
    } else {
        auto rAux = getReg64();
        Xbyak::Reg32 r32Aux = Xbyak::Reg32(rAux.getIdx());
        const uint8_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / sizeof(int);

        for (uint8_t i = 0; i < elPerVec; i++) {
            Xbyak::Label lLoopNext;
            if (useMask) {
                uni_vpextrd(r32Aux, vReadMask, i);
                cmp(r32Aux, 0); // TODO: check significant bit
                je(lLoopNext, T_NEAR);
            }
            uni_vpextrd(r32Aux, vSrcShift, i);
            pinsrd(vDst, ptr[rSrcPtr + rAux], i);

            if (useMask)
                L(lLoopNext);
        }
    }
}

void jit_kernel_base::gatherdd(const Xbyak::Ymm&   vDst,
                             const Xbyak::Reg64& rSrcPtr,
                             const Xbyak::Ymm&   vSrcShift,
                             const Xbyak::Ymm&   vReadMask,
                             const bool useMask,
                             const bool zeroFill) {
    if (vDst.getIdx() == vSrcShift.getIdx() || vDst.getIdx() == vReadMask.getIdx() || vSrcShift.getIdx() == vReadMask.getIdx()) {
        IE_THROW() << "Any pair of the index, mask, or destination registers cannot be the same.";
    }

    if (zeroFill)
        uni_vpxor(vDst, vDst, vDst);

    if (is_valid_isa(x64::avx2)) {
        if (!useMask)
            uni_vpcmpeqd(vReadMask, vReadMask, vReadMask);

        vpgatherdd(vDst, ptr[rSrcPtr + vSrcShift], vReadMask);
    } else {
        Xbyak::Xmm xmmDst      = Xbyak::Xmm(vDst.getIdx()),
                   xmmSrcShft  = Xbyak::Xmm(vSrcShift.getIdx()),
                   xmmReadMask = Xbyak::Xmm(vReadMask.getIdx());
        for (uint8_t i = 0; i < 2; i++) {
            gatherdd(xmmDst, rSrcPtr, xmmSrcShft, xmmReadMask, useMask, false);

            vperm2f128(vDst, vDst, vDst, 0x1);
            vperm2f128(vSrcShift, vSrcShift, vSrcShift, 0x1);
            if (useMask)
                vperm2f128(vReadMask, vReadMask, vReadMask, 0x1);
        }
    }
}

void jit_kernel_base::fillRestWorkMask(const Xbyak::Opmask& dstMask,
                                     const Xbyak::Zmm&    zAux,
                                     const Xbyak::Reg64&  rWorkRest) {
    auto rAux0 = getReg64();
    auto rAux1 = getReg64();
    Xbyak::Label lKmov;
    Xbyak::Reg32 rOnes(rAux1.getIdx());
    const uint64_t typeSize = 4;
    const uint64_t elPerVec = x64::cpu_isa_traits<x64::avx512_core>::vlen / typeSize;

    mov(rOnes, 0x0000FFFF);
    cmp(rWorkRest, elPerVec);
    jge(lKmov);
    {
        Xbyak::Reg32 rShift(rAux0.getIdx());
        mov(rShift, elPerVec);
        sub(rShift, rWorkRest);
        shrx(rOnes, rOnes, rShift);
    }
    L(lKmov);
    kmovw(dstMask, rOnes);
}

void jit_kernel_base::memMovDD(const Xbyak::Reg64& rDst,
                             const Xbyak::Reg64& rSrc,
                             const Xbyak::Xmm&   vReadMask,
                             const Xbyak::Xmm&   vSrcShift,
                             const Xbyak::Reg64& rToStoreNum,
                             const bool          useMask,
                             const bool          zeroFill) {
    Xbyak::Label lEnd;
    auto rAux = getReg64();
    Xbyak::Reg32 r32Aux = Xbyak::Reg32(rAux.getIdx());
    const uint8_t typeSize = sizeof(int);
    const uint8_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;

    for (uint8_t i = 0; i < elPerVec; i++) {
        cmp(rToStoreNum, i);
        jle(lEnd, T_NEAR);

        Xbyak::Label lLoopNext;
        if (useMask) {
            uni_vpextrd(r32Aux, vReadMask, i);
            cmp(r32Aux, 0);
            if (zeroFill) {
                Xbyak::Label lNotZero;
                jne(lNotZero, T_NEAR);
                mov(ptr[rDst.getReg() + i * typeSize], r32Aux);
                jmp(lLoopNext, T_NEAR);
                L(lNotZero);
            } else {
                je(lLoopNext, T_NEAR);
            }
        }
        uni_vpextrd(r32Aux, vSrcShift, i);
        mov(r32Aux, ptr[rSrc.getReg() + rAux]);
        mov(ptr[rDst.getReg() + i * typeSize], r32Aux);

        L(lLoopNext);
    }
    L(lEnd);
}

void jit_kernel_base::memMovDD(const Xbyak::Reg64& rDst,
                             const Xbyak::Reg64& rSrc,
                             const Xbyak::Ymm&   vReadMask,
                             const Xbyak::Ymm&   vSrcShift,
                             const Xbyak::Reg64& rToStoreNum,
                             const bool          useMask,
                             const bool          zeroFill) {
    Xbyak::Label lEnd;
    if (is_valid_isa(x64::avx2)) {
        auto vAux = RegistersPool::Reg<Xbyak::Ymm>(registersPool);
        gatherdd(vAux, rSrc, vSrcShift, vReadMask, useMask, zeroFill);
        store(ptr[rDst], vAux, rToStoreNum, sizeof(int));
    } else if (is_valid_isa(x64::avx)) {
        const uint8_t typeSize = sizeof(int);
        const uint8_t elPerXmm = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;
        Xbyak::Xmm xmmReadMask  = Xbyak::Xmm(vReadMask.getIdx()),
                   xmmSrcShft   = Xbyak::Xmm(vSrcShift.getIdx());
        for (uint8_t i = 0; i < 2; i++) {
            memMovDD(rDst, rSrc, xmmReadMask, xmmSrcShft, rToStoreNum, useMask, zeroFill);

            if (i == 0) {
                cmp(rToStoreNum, elPerXmm);
                jle(lEnd, T_NEAR);
                sub(rToStoreNum, elPerXmm);
                add(rDst, typeSize * elPerXmm);
            } else {
                add(rToStoreNum, elPerXmm);
                sub(rDst, typeSize * elPerXmm);
            }

            vperm2f128(vSrcShift, vSrcShift, vSrcShift, 0x1);
            if (useMask)
                vperm2f128(vReadMask, vReadMask, vReadMask, 0x1);
        }
    }
    L(lEnd);
}

void jit_kernel_base::createRegistersPool() {
    registersPool = RegistersPool::create(max_cpu_isa_, {abi_param1});
}

void jit_kernel_base::createStackAllocator() {
    stackAllocator = std::unique_ptr<StackAllocator>(new StackAllocator{*this});
}
