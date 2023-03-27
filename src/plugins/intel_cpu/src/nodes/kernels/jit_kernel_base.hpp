// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"
#include "registers_pool.hpp"

namespace ov {
namespace intel_cpu {

#define getReg64() RegistersPool::Reg<Xbyak::Reg64>(registersPool)
#define getReg32() RegistersPool::Reg<Xbyak::Reg32>(registersPool)
#define getVmm()   RegistersPool::Reg<Vmm>(registersPool)
#define getMask()  RegistersPool::Reg<Vmask>(registersPool)

class JitKernelBase: public dnnl::impl::cpu::x64::jit_generator {
public:
    JitKernelBase(const char* name) : dnnl::impl::cpu::x64::jit_generator(name) {}

    void uni_vfmsub132ps(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrc, const Xbyak::Operand& op);

    void uni_vfnmadd132ps(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrc, const Xbyak::Operand& op);

    void uni_vfmsub231ps(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrc, const Xbyak::Operand& op);

    void uni_vpaddd(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrc, const Xbyak::Operand& op) {
        jit_generator::uni_vpaddd(vDst, vSrc, op);
    }

    void uni_vpaddd(const Xbyak::Ymm& vDst, const Xbyak::Ymm& vSrc, const Xbyak::Operand& op);

    void uni_vpsubd(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrc, const Xbyak::Operand& op) {
        jit_generator::uni_vpsubd(vDst, vSrc, op);
    }

    void uni_vpsubd(const Xbyak::Ymm& vDst, const Xbyak::Ymm& vSrc, const Xbyak::Operand& op);

    void uni_vdivps(const Xbyak::Xmm& vDst, const Xbyak::Operand& op1, const Xbyak::Operand& op2);

    void uni_vandps(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrs, const Xbyak::Operand &op);

    void uni_vandnps(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrs, const Xbyak::Operand &op);

    void uni_kmovd(const Xbyak::Opmask& kDst, const Xbyak::Opmask& kSrc) {
        kmovd(kDst, kSrc);
    }

    void uni_kmovd(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrc) {
        uni_vmovups(vDst, vSrc);
    }

    void uni_kandd(const Xbyak::Opmask& kDst, const Xbyak::Opmask& kSrc1, const Xbyak::Opmask& kSrc2) {
        kandd(kDst, kSrc1, kSrc2);
    }

    void uni_kandd(const Xbyak::Xmm& kDst, const Xbyak::Xmm& kSrc1, const Xbyak::Xmm& kSrc2) {
        uni_vandps(kDst, kSrc1, kSrc2);
    }

    void uni_vpbroadcastd(const Xbyak::Xmm &x, const Xbyak::Operand &op);

    void uni_vpbroadcastd(const Xbyak::Ymm &x, const Xbyak::Operand &op);

    void gatherdd(const Xbyak::Xmm&    vDst,
                  const Xbyak::Reg64&  rSrcPtr,
                  const Xbyak::Xmm&    vSrcShift,
                  const Xbyak::Opmask& kReadMask,
                  const bool useMask   = true,
                  const bool zeroFill  = false);

    void gatherdd(const Xbyak::Xmm&   vDst,
                  const Xbyak::Reg64& rSrcPtr,
                  const Xbyak::Xmm&   vSrcShift,
                  const Xbyak::Xmm&   vReadMask,
                  const bool useMask  = true,
                  const bool zeroFill = false);

    void gatherdd(const Xbyak::Ymm&   vDst,
                  const Xbyak::Reg64& rSrcPtr,
                  const Xbyak::Ymm&   vSrcShift,
                  const Xbyak::Ymm&   vReadMask,
                  const bool useMask  = true,
                  const bool zeroFill = false);

    void fillRestWorkMask(const Xbyak::Opmask& kDstMask,
                          const Xbyak::Reg64& rWorkRest);

    void fillRestWorkMask(const Xbyak::Xmm& ymmDstMask,
                          const Xbyak::Reg64& rWorkRest,
                          const uint64_t typeSize = 4);

    void fillRestWorkMask(const Xbyak::Ymm& ymmDstMask,
                          const Xbyak::Reg64& rWorkRest,
                          const uint64_t typeSize = 4);

    void load(const Xbyak::Xmm&     vDst,
              const Xbyak::Address& srcAddr,
              const Xbyak::Reg64&   rLoadNum,
              const size_t          typeSize,
              const bool zeroFill = false);

    void load(const Xbyak::Ymm&     vDst,
              const Xbyak::Address& srcAddr,
              const Xbyak::Reg64&   rLoadNum,
              const size_t          typeSize,
              const bool zeroFill = false);

    void store(const Xbyak::Address& dstAddr,
               const Xbyak::Xmm&     vSrc,
               const Xbyak::Reg64&   rToStoreNum,
               const size_t          typeSize);

    void store(const Xbyak::Address& dstAddr,
               const Xbyak::Ymm&     vSrc,
               const Xbyak::Reg64&   rToStoreNum,
               const size_t          typeSize);

    // Makes gather from memory under the vReadMask and writes to the memory m128.
    void memMovDD(const Xbyak::Reg64& rDst,
                  const Xbyak::Reg64& rSrc,
                  const Xbyak::Xmm&   vReadMask,
                  const Xbyak::Xmm&   vSrcShift,
                  const Xbyak::Reg64& rToStoreCounter,
                  const bool useMask  = true,
                  const bool zeroFill = false);

    // Makes gather from the memory under the vReadMask and writes to the memory m256.
    void memMovDD(const Xbyak::Reg64& rDst,
                  const Xbyak::Reg64& rSrc,
                  const Xbyak::Ymm&   vReadMask,
                  const Xbyak::Ymm&   vSrcShift,
                  const Xbyak::Reg64& rToStoreCounter,
                  const bool useMask  = true,
                  const bool zeroFill = false);

protected:
    inline bool isValidIsa(dnnl::impl::cpu::x64::cpu_isa_t isa) {
        return dnnl::impl::cpu::x64::mayiuse(isa);
    }

    RegistersPool::Ptr registersPool;

    enum {
        // Comparison predicate operand (immediate byte) for single-precision floating-point values.
        CMP_EQ_PS = 0, // Equal (ordered, non-signaling)
        CMP_LT_PS,     // Less-than (ordered, signaling)
        CMP_LE_PS,     // Less-than-or-equal (ordered, signaling)
        CMP_UNORD_PS,  // Unordered (non-signaling)
        CMP_NEQ_PS,    // Not-equal (unordered, non-signaling)
        CMP_NLT_PS,    // Not-less-than (unordered, signaling)
        CMP_NLE_PS,    // Not-less-than-or-equal (unordered, signaling)
        CMP_ORD_PS     // Ordered (non-signaling)
    };
};

} // namespace intel_cpu
} // namespace ov
