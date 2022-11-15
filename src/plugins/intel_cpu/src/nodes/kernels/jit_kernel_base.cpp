// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_base.hpp"

using namespace ov;
using namespace intel_cpu;
using namespace dnnl::impl::cpu;


void JitKernelBase::generate() {
    this->preamble();

    createRegistersPool();
    stackAllocator = std::unique_ptr<StackAllocator>(new StackAllocator{*this});

    generate_impl();

    registersPool.reset();
    stackAllocator.reset();

    this->postamble();

    for (auto& record : emittersMap) {
        record.second->emit_data();
    }
}

void JitKernelBase::uni_vfmsub132ps(const Xbyak::Xmm& vDst,
                                    const Xbyak::Xmm& vSrc,
                                    const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vfmsub132ps(vDst, vSrc, op);
    } else if (isValidIsa(x64::avx)) {
        assert(vDst.getIdx() != vSrc.getIdx());
        vmulps(vDst, vDst, op);
        vsubps(vDst, vDst, vSrc);
    } else {
        assert(vDst.getIdx() != vSrc.getIdx());
        mulps(vDst, op);
        subps(vDst, vSrc);
    }
}

void JitKernelBase::uni_vfnmadd132ps(const Xbyak::Xmm& vDst,
                                     const Xbyak::Xmm& vSrc,
                                     const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vfnmadd132ps(vDst, vSrc, op);
    } else if (isValidIsa(x64::avx)) {
        assert(vDst.getIdx() != vSrc.getIdx());
        vmulps(vDst, vDst, op);
        vsubps(vDst, vSrc, vDst);
    } else {
        assert(vDst.getIdx() != vSrc.getIdx());
        mulps(vDst, op);
        subps(vSrc, vDst);
        movups(vDst, vSrc);
    }
}

void JitKernelBase::uni_vfmsub231ps(const Xbyak::Xmm& vDst,
                                    const Xbyak::Xmm& vSrc,
                                    const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vfmsub231ps(vDst, vSrc, op);
    } else if (isValidIsa(x64::avx)) {
        assert(!vDst.isEqualIfNotInherited(op));
        vmulps(vSrc, vSrc, op);
        vsubps(vDst, vSrc, vDst);
    } else {
        assert(!vDst.isEqualIfNotInherited(op));
        mulps(vSrc, op);
        subps(vSrc, vDst);
        movups(vDst, vSrc);
    }
}

void JitKernelBase::uni_vpaddd(const Xbyak::Ymm& vDst,
                               const Xbyak::Ymm& vSrc,
                               const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vpaddd(vDst, vSrc, op);
    } else if (isValidIsa(x64::avx)) {
        Xbyak::Xmm xmmDst(vDst.getIdx());
        vmovups(vDst, vSrc);
        if (op.isYMM()) {
            Xbyak::Ymm ymmOp(op.getIdx());
            Xbyak::Xmm xmmOp(op.getIdx());
            paddd(xmmDst, xmmOp);
            vperm2f128(vDst, vDst, vDst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
            paddd(xmmDst, xmmOp);
            vperm2f128(vDst, vDst, vDst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
        } else if (op.isMEM()) {
            const int vlen = x64::cpu_isa_traits<x64::sse41>::vlen;
            paddd(xmmDst, op.getAddress());
            vperm2f128(vDst, vDst, vDst, 0x1);
            paddd(xmmDst, ptr[op.getAddress().getRegExp() + vlen]);
            vperm2f128(vDst, vDst, vDst, 0x1);
        } else {
            IE_THROW() << "Not supported operand type.";
        }
    } else if (isValidIsa(x64::sse41)) {
        assert(vDst.getIdx() != vSrc.getIdx());
        paddd(vDst, op);
    } else {
        IE_THROW() << "Not defined behavior for instruction 'vpaddd' in current instructions set.";
    }
}

void JitKernelBase::uni_vpsubd(const Xbyak::Ymm& vDst,
                               const Xbyak::Ymm& vSrc,
                               const Xbyak::Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vpsubd(vDst, vSrc, op);
    } else if (isValidIsa(x64::avx)) {
        Xbyak::Xmm xmmDst(vDst.getIdx());
        vmovups(vDst, vSrc);
        if (op.isYMM()) {
            Xbyak::Ymm ymmOp(op.getIdx());
            Xbyak::Xmm xmmOp(op.getIdx());
            psubd(xmmDst, xmmOp);
            vperm2f128(vDst, vDst, vDst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
            psubd(xmmDst, xmmOp);
            vperm2f128(vDst, vDst, vDst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
        } else if (op.isMEM()) {
            const int vlen = x64::cpu_isa_traits<x64::sse41>::vlen;
            psubd(xmmDst, op.getAddress());
            vperm2f128(vDst, vDst, vDst, 0x1);
            psubd(xmmDst, ptr[op.getAddress().getRegExp() + vlen]);
            vperm2f128(vDst, vDst, vDst, 0x1);
        } else {
            IE_THROW() << "Not supported operand type.";
        }
    } else if (isValidIsa(x64::sse41)) {
        assert(vDst.getIdx() != vSrc.getIdx());
        psubd(vDst, op);
    } else {
        IE_THROW() << "Not defined behavior for instruction 'vpsubd' in current instructions set.";
    }
}

void JitKernelBase::uni_vdivps(const Xbyak::Xmm& vDst,
                               const Xbyak::Operand& op1,
                               const Xbyak::Operand& op2) {
    if (isValidIsa(x64::avx)) {
        vdivps(vDst, op1, op2);
    } else {
        if (!vDst.isEqualIfNotInherited(op1)) {
            movups(vDst, op1);
        }
        divps(vDst, op2);
    }
}

void JitKernelBase::uni_vandps(const Xbyak::Xmm& vDst,
                               const Xbyak::Xmm& vSrs,
                               const Xbyak::Operand &op) {
    if (isValidIsa(x64::avx)) {
        vandps(vDst, vSrs, op);
    } else {
        if (!vDst.isEqualIfNotInherited(vSrs)) {
            movups(vDst, vSrs);
        }
        andps(vDst, op);
    }
}

void JitKernelBase::uni_vandnps(const Xbyak::Xmm& vDst,
                                const Xbyak::Xmm& vSrs,
                                const Xbyak::Operand &op) {
    if (isValidIsa(x64::avx)) {
        vandnps(vDst, vSrs, op);
    } else {
        if (!vDst.isEqualIfNotInherited(vSrs)) {
            movups(vDst, vSrs);
        }
        andnps(vDst, op);
    }
}

void JitKernelBase::emu_vgatherdps(const Xbyak::Xmm& xmm_val,
                                   const Xbyak::Reg64& reg_addr,
                                   const Xbyak::Xmm& xmm_index,
                                   const int& scale,
                                   const int& disp,
                                   const Xbyak::Reg& reg_mask,
                                   const bool is_mask_seq/* = true*/) {
    const size_t kDataTypeSize = sizeof(float);
    Xbyak::Xmm xmm_mask{reg_mask.getIdx(), reg_mask.getKind(), static_cast<int>(reg_mask.getBit())};
    std::vector<Xbyak::Xmm> not_available_xmm{xmm_index, xmm_val, xmm_mask};
    if (isValidIsa(x64::avx512_core)) {
        const Xbyak::Zmm zmm_zero_val = registersPool->getInplaceFree<Xbyak::Zmm>(not_available_xmm);
        push(zmm_zero_val);
        uni_vxorps(zmm_zero_val, zmm_zero_val, zmm_zero_val);
        RegistersPool::Reg<Xbyak::Opmask> avx512_mask{registersPool, 1};
        vpcmpud(avx512_mask, Xbyak::Zmm{reg_mask.getIdx()}, zmm_zero_val, VCMPPS_GT);
        pop(zmm_zero_val);
        vgatherdps(xmm_val | avx512_mask, ptr[reg_addr + xmm_index * scale + disp]);
    } else if (isValidIsa(x64::avx2)) {
        assert(reg_mask.isYMM());
        Xbyak::Ymm ymm_mask{reg_mask.getIdx()};
        vgatherdps(xmm_val, ptr[reg_addr + xmm_index * scale + disp], ymm_mask);
    } else {
        const size_t kSimdWidth = x64::cpu_isa_traits<x64::sse41>::vlen / kDataTypeSize;
        assert(reg_mask.isXMM());
        assert(xmm_val.getKind() == xmm_index.getKind());
        assert(xmm_index.getKind() == xmm_mask.getKind());

        std::vector<Xbyak::Reg> not_available_reg{reg_addr};
        const Xbyak::Reg64 idx = registersPool->getInplaceFree<Xbyak::Reg64>(not_available_reg);
        const Xbyak::Reg64 mask = registersPool->getInplaceFree<Xbyak::Reg64>(not_available_reg);

        push(idx);
        push(mask);
        xor_(idx, idx);
        xor_(mask, mask);

        Xbyak::Label gather_fast_end;
        for (int i = 0; i < static_cast<int>(kSimdWidth); i++) {
            Xbyak::Label gather_end;
            uni_vpextrd(mask.cvt32(), xmm_mask, i);
            cmp(mask.cvt32(), 0xFFFFFFFF);
            if (is_mask_seq) {
                jne(gather_fast_end, T_SHORT);
            } else {
                jne(gather_end, T_SHORT);
            }
            uni_vpextrd(idx.cvt32(), xmm_index, i);
            Xbyak::Address addr = ptr[reg_addr + idx * scale + disp];
            uni_vpinsrd(xmm_val, xmm_val, addr, i);
            if (!is_mask_seq) {
                L(gather_end);
            }
        }
        if (is_mask_seq) {
            L(gather_fast_end);
        }

        pop(mask);
        pop(idx);
    }
}

void JitKernelBase::emu_vscatterdps(const Xbyak::Reg64& reg_addr,
                                    const Xbyak::Xmm& xmm_index,
                                    const int scale,
                                    const int disp,
                                    const Xbyak::Xmm& xmm_val,
                                    const Xbyak::Reg& reg_mask,
                                    const bool is_mask_seq /* = true*/) {
    const size_t kDataTypeSize = sizeof(float);
    Xbyak::Xmm xmm_mask{reg_mask.getIdx(), reg_mask.getKind(), static_cast<int>(reg_mask.getBit())};
    std::vector<Xbyak::Xmm> not_available_xmm{xmm_index, xmm_val, xmm_mask};
    if (isValidIsa(x64::avx512_core)) {
        const Xbyak::Zmm zmm_zero_val = registersPool->getInplaceFree<Xbyak::Zmm>(not_available_xmm);
        push(zmm_zero_val);
        uni_vxorps(zmm_zero_val, zmm_zero_val, zmm_zero_val);
        RegistersPool::Reg<Xbyak::Opmask> avx512_mask{registersPool, 1};
        vpcmpud(avx512_mask, Xbyak::Zmm{reg_mask.getIdx()}, zmm_zero_val, VCMPPS_GT);
        pop(zmm_zero_val);
        vscatterdps(ptr[reg_addr + xmm_index * scale + disp], xmm_val | avx512_mask);
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
        if (isValidIsa(x64::avx2)) {
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
                Xbyak::Address addr = ptr[reg_addr + idx * scale];
                uni_vpextrd(val.cvt32(), xmm_val, i);
                mov(addr, val.cvt32());
                if (!is_mask_seq) {
                    L(scatter_end);
                }
            }
        };

        if (isValidIsa(x64::avx2)) {
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

        if (isValidIsa(x64::avx2)) {
            pop(Xbyak::Ymm{xmm_val_temp.getIdx()});
            pop(Xbyak::Ymm{xmm_index_temp.getIdx()});
            pop(Xbyak::Ymm{xmm_mask_temp.getIdx()});
        }
        pop(val);
        pop(mask);
        pop(idx);
    }
}

void JitKernelBase::gatherdd(const Xbyak::Xmm&    vDst,
                             const Xbyak::Reg64&  rSrcPtr,
                             const Xbyak::Xmm&    vSrcShift,
                             const Xbyak::Opmask& kReadMask,
                             const bool useMask,
                             const bool zeroFill) {
    if (kReadMask.getIdx() == 0) {
        IE_THROW() << "The vpgatherdd instruction cannot use the register k0 as mask.";
    }
    if (!useMask)
        kxnord(kReadMask, kReadMask, kReadMask);
    if (zeroFill)
        uni_vpxor(vDst, vDst, vDst);

    vpgatherdd(vDst | kReadMask, ptr[rSrcPtr + vSrcShift]);
}

void JitKernelBase::gatherdd(const Xbyak::Xmm&   vDst,
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

    if (isValidIsa(x64::avx2)) {
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

void JitKernelBase::gatherdd(const Xbyak::Ymm&   vDst,
                             const Xbyak::Reg64& rSrcPtr,
                             const Xbyak::Ymm&   vSrcShift,
                             const Xbyak::Ymm&   vReadMask,
                             const bool useMask,
                             const bool zeroFill) {
    if (vDst.getIdx() == vSrcShift.getIdx() || vDst.getIdx() == vReadMask.getIdx() || vSrcShift.getIdx() == vReadMask.getIdx()) {
        IE_THROW() << "Any pair of the index, mask, or destination registers cannot be the same.";
    }
    if (isValidIsa(x64::avx2)) {
        if (!useMask)
            uni_vpcmpeqd(vReadMask, vReadMask, vReadMask);
        if (zeroFill)
            uni_vpxor(vDst, vDst, vDst);

        vpgatherdd(vDst, ptr[rSrcPtr + vSrcShift], vReadMask);
    } else {
        Xbyak::Xmm xmmDst      = Xbyak::Xmm(vDst.getIdx()),
                   xmmSrcShft  = Xbyak::Xmm(vSrcShift.getIdx()),
                   xmmReadMask = Xbyak::Xmm(vReadMask.getIdx());
        for (uint8_t i = 0; i < 2; i++) {
            gatherdd(xmmDst, rSrcPtr, xmmSrcShft, xmmReadMask, useMask, zeroFill);

            vperm2f128(vDst, vDst, vDst, 0x1);
            vperm2f128(vSrcShift, vSrcShift, vSrcShift, 0x1);
            if (useMask)
                vperm2f128(vReadMask, vReadMask, vReadMask, 0x1);
        }
    }
}

void JitKernelBase::uni_vpbroadcastd(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
    if (isValidIsa(x64::avx2)) {
        vpbroadcastd(x, op);
    } else if (isValidIsa(x64::avx)) {
        if (op.isMEM()) {
            vbroadcastss(x, op.getAddress());
        } else {
            vmovss(x, x, op);
            vpshufd(x, x, 0x0);
        }
    } else {
        movss(x, op);
        pshufd(x, x, 0x0);
    }
}

void JitKernelBase::uni_vpbroadcastd(const Xbyak::Ymm &x, const Xbyak::Operand &op) {
    if (isValidIsa(x64::avx2)) {
        vpbroadcastd(x, op);
    } else {
        if (op.isMEM()) {
            vbroadcastss(x, op.getAddress());
        } else {
            const Xbyak::Xmm t(x.getIdx());
            if (!t.isEqualIfNotInherited(op)) {
                vmovss(t, t, op);
            }
            vinsertf128(x, x, t, 1);
            vshufps(x, x, x, 0);
        }
    }
}

void JitKernelBase::uni_vaddps(const Xbyak::Xmm& x, const Xbyak::Xmm& op1, const Xbyak::Operand& op2) {
    if (isValidIsa(x64::avx)) {
        vaddps(x, op1, op2);
    } else {
        if (x.getIdx() == op1.getIdx()) {
            addps(x, op2);
        } else if (x.isEqualIfNotInherited(op2)) {
            addps(x, op1);
        } else {
            movups(x, op1);
            addps(x, op2);
        }
    }
}

void JitKernelBase::uni_vsubps(const Xbyak::Xmm& x, const Xbyak::Xmm& op1, const Xbyak::Operand& op2) {
    if (isValidIsa(x64::avx)) {
        vsubps(x, op1, op2);
    } else {
        if (x.getIdx() == op1.getIdx()) {
            subps(x, op2);
        } else if (x.isEqualIfNotInherited(op2)) {
            push(op1);
            subps(op1, op2);
            movups(x, op1);
            pop(op1);
        } else {
            movups(x, op1);
            subps(x, op2);
        }
    }
}

void JitKernelBase::uni_vcmpps(const Xbyak::Xmm& x,
                               const Xbyak::Xmm& op1,
                               const Xbyak::Operand& op2,
                               const int cmp_predicate) {
    if (isValidIsa(x64::avx)) {
        vcmpps(x, op1, op2, cmp_predicate);
    } else {
        if (x.getIdx() == op1.getIdx()) {
            cmpps(x, op2, cmp_predicate);
        } else if (x.isEqualIfNotInherited(op2)) {
            push(op1);
            cmpps(op1, op2, cmp_predicate);
            movups(x, op1);
            pop(op1);
        } else {
            movups(x, op1);
            cmpps(x, op2, cmp_predicate);
        }
    }
}

void JitKernelBase::fillRestWorkMask(const Xbyak::Opmask& dstMask,
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

void JitKernelBase::load(const Xbyak::Xmm&     vDst,
                         const Xbyak::Address& srcAddr,
                         const Xbyak::Reg64&   rLoadNum,
                         const size_t          typeSize,
                         const bool            zeroFilling) {
    if (!one_of(typeSize, 1, 2, 4, 8)) {
        IE_THROW() << "Could not load data with type size " << typeSize;
    }
    const uint8_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;
    Xbyak::Label lEnd;
    if (zeroFilling)
        pxor(vDst, vDst);

    for (uint8_t i = 0; i < elPerVec; i++) {
        cmp(rLoadNum, i);
        jle(lEnd, T_NEAR);

        const size_t offset = i * typeSize;
        if (typeSize == 1)
            pinsrb(vDst, ptr[srcAddr.getRegExp() + offset], i);
        else if (typeSize == 2)
            pinsrw(vDst, ptr[srcAddr.getRegExp() + offset], i);
        else if (typeSize == 4)
            pinsrd(vDst, ptr[srcAddr.getRegExp() + offset], i);
        else if (typeSize == 8)
            pinsrq(vDst, ptr[srcAddr.getRegExp() + offset], i);
    }
    L(lEnd);
}

void JitKernelBase::load(const Xbyak::Ymm&     vDst,
                         const Xbyak::Address& srcAddr,
                         const Xbyak::Reg64&   rLoadNum,
                         const size_t          typeSize,
                         const bool            zeroFilling) {
    if (!one_of(typeSize, 1, 2, 4, 8)) {
        IE_THROW() << "Could not load data with type size " << typeSize;
    }
    const size_t elPerXmm = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;
    Xbyak::Label lEnd;
    if (zeroFilling)
        uni_vpxor(vDst, vDst, vDst);
    Xbyak::Xmm xmmDst(vDst.getIdx());

    for (size_t i = 0lu; i < 2lu; i++) {
        Xbyak::Label lPerm;
        const size_t idx = i * elPerXmm;
        const size_t offset0 = idx * typeSize;

        for (size_t j = 0lu; j < elPerXmm; j++) {
            cmp(rLoadNum, j + idx);
            jle(i == 0 ? lEnd : lPerm, T_NEAR);

            const size_t offset = offset0 + j * typeSize;
            if (typeSize == 1)
                pinsrb(xmmDst, ptr[srcAddr.getRegExp() + offset], j);
            else if (typeSize == 2)
                pinsrw(xmmDst, ptr[srcAddr.getRegExp() + offset], j);
            else if (typeSize == 4)
                pinsrd(xmmDst, ptr[srcAddr.getRegExp() + offset], j);
            else if (typeSize == 8)
                pinsrq(xmmDst, ptr[srcAddr.getRegExp() + offset], j);
        }

        L(lPerm);
        vperm2f128(vDst, vDst, vDst, 0x1);
    }
    L(lEnd);
}

void JitKernelBase::store(const Xbyak::Address& dstAddr,
                          const Xbyak::Xmm&     vSrc,
                          const Xbyak::Reg64&   rToStoreNum,
                          const size_t          typeSize) {
    if (!one_of(typeSize, 1, 2, 4, 8)) {
        IE_THROW() << "Could not store data with type size " << typeSize;
    }
    Xbyak::Label lEnd;
    const size_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;

    for (size_t i = 0; i < elPerVec; i++) {
        cmp(rToStoreNum, i);
        jle(lEnd, T_NEAR);

        const size_t offset = i * typeSize;
        if (typeSize == 1) {
            uni_vpextrb(ptr[dstAddr.getRegExp() + offset], vSrc, i);
        } else if (typeSize == 2) {
            uni_vpextrw(ptr[dstAddr.getRegExp() + offset], vSrc, i);
        } else if (typeSize == 4) {
            uni_vpextrd(ptr[dstAddr.getRegExp() + offset], vSrc, i);
        } else if (typeSize == 8) {
            uni_vpextrq(ptr[dstAddr.getRegExp() + offset], vSrc, i);
        }
    }
    L(lEnd);
}

void JitKernelBase::store(const Xbyak::Address& dstAddr,
                          const Xbyak::Ymm&     vSrc,
                          const Xbyak::Reg64&   rToStoreNum,
                          const size_t          typeSize) {
    if (!one_of(typeSize, 1, 2, 4, 8)) {
        IE_THROW() << "Could not store data with type size " << typeSize;
    }
    Xbyak::Label lEnd;
    Xbyak::Xmm xmmSrc(vSrc.getIdx());
    const size_t elPerXmm = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;

    for (int i = 0; i < 2; i++) {
        Xbyak::Label lPerm;
        const size_t idx = i * elPerXmm;
        const size_t offset0 = idx * typeSize;

        for (size_t j = 0; j < elPerXmm; j++) {
            cmp(rToStoreNum, j + idx);
            jle(i == 0 ? lEnd : lPerm, T_NEAR);

            const size_t offset = offset0 + j * typeSize;
            if (typeSize == 8) {
                uni_vpextrq(ptr[dstAddr.getRegExp() + offset], xmmSrc, j);
            } else if (typeSize == 4) {
                uni_vpextrd(ptr[dstAddr.getRegExp() + offset], xmmSrc, j);
            } else if (typeSize == 2) {
                uni_vpextrw(ptr[dstAddr.getRegExp() + offset], xmmSrc, j);
            } else if (typeSize == 1) {
                uni_vpextrb(ptr[dstAddr.getRegExp() + offset], xmmSrc, j);
            }
        }

        L(lPerm);
        vperm2f128(vSrc, vSrc, vSrc, 0x1);
    }
    L(lEnd);
}


void JitKernelBase::push(const Xbyak::Xmm& xmm) {
    if (xmm.isXMM()) {
        sub(rsp, xmm_len);
        uni_vmovdqu(ptr[rsp], xmm);
    } else if (xmm.isYMM()) {
        sub(rsp, ymm_len);
        uni_vmovdqu(ptr[rsp], Xbyak::Ymm{xmm.getIdx()});
    } else if (xmm.isZMM()) {
        sub(rsp, zmm_len);
        uni_vmovdqu(ptr[rsp], Xbyak::Zmm{xmm.getIdx()});
    }
}

void JitKernelBase::pop(const Xbyak::Xmm& xmm) {
    if (xmm.isXMM()) {
        uni_vmovdqu(xmm, ptr[rsp]);
        add(rsp, xmm_len);
    } else if (xmm.isYMM()) {
        uni_vmovdqu(Xbyak::Ymm{xmm.getIdx()}, ptr[rsp]);
        add(rsp, ymm_len);
    } else if (xmm.isZMM()) {
        uni_vmovdqu(Xbyak::Zmm{xmm.getIdx()}, ptr[rsp]);
        add(rsp, zmm_len);
    }
}


void JitKernelBase::memMovDD(const Xbyak::Reg64& rDst,
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

void JitKernelBase::memMovDD(const Xbyak::Reg64& rDst,
                             const Xbyak::Reg64& rSrc,
                             const Xbyak::Ymm&   vReadMask,
                             const Xbyak::Ymm&   vSrcShift,
                             const Xbyak::Reg64& rToStoreNum,
                             const bool          useMask,
                             const bool          zeroFill) {
    Xbyak::Label lEnd;
    if (isValidIsa(x64::avx2)) {
        auto vAux = RegistersPool::Reg<Xbyak::Ymm>(registersPool);
        gatherdd(vAux, rSrc, vSrcShift, vReadMask, useMask, zeroFill);
        store(ptr[rDst], vAux, rToStoreNum, sizeof(int));
    } else if (isValidIsa(x64::avx)) {
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
