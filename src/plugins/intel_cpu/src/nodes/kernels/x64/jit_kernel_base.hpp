// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"
#include "registers_pool.hpp"
#include "emitters/x64/jit_bf16_emitters.hpp"

namespace ov {
namespace intel_cpu {
namespace kernel {

#define getReg64() RegistersPool::Reg<Xbyak::Reg64>(this->registersPool)
#define getReg32() RegistersPool::Reg<Xbyak::Reg32>(this->registersPool)
#define getVmm()   RegistersPool::Reg<Vmm>(this->registersPool)
#define getMask()  RegistersPool::Reg<Vmask>(this->registersPool)

class JitKernelBase: public dnnl::impl::cpu::x64::jit_generator {
public:
    JitKernelBase(const char* name, dnnl::impl::cpu::x64::cpu_isa_t max_cpu_isa);

    void uni_vfmsub132ps(const Xbyak::Xmm& vmm_dst, const Xbyak::Xmm& vmm_src, const Xbyak::Operand& op);

    void uni_vfnmadd132ps(const Xbyak::Xmm& vmm_dst, const Xbyak::Xmm& vmm_src, const Xbyak::Operand& op);

    void uni_vfmsub231ps(const Xbyak::Xmm& vmm_dst, const Xbyak::Xmm& vmm_src, const Xbyak::Operand& op);

    void uni_vpaddd(const Xbyak::Xmm& vmm_dst, const Xbyak::Xmm& vmm_src, const Xbyak::Operand& op) {
        jit_generator::uni_vpaddd(vmm_dst, vmm_src, op);
    }

    void uni_vpaddd(const Xbyak::Ymm& vmm_dst, const Xbyak::Ymm& vmm_src, const Xbyak::Operand& op);

    void uni_vaddpd(const Xbyak::Xmm& vmm_dst, const Xbyak::Operand &op1, const Xbyak::Operand &op2);

    void uni_vpsubd(const Xbyak::Xmm& vmm_dst, const Xbyak::Xmm& vmm_src, const Xbyak::Operand& op) {
        jit_generator::uni_vpsubd(vmm_dst, vmm_src, op);
    }

    void uni_vpsubd(const Xbyak::Ymm& vmm_dst, const Xbyak::Ymm& vmm_src, const Xbyak::Operand& op);

    void uni_vmulpd(const Xbyak::Xmm& vmm_dst, const Xbyak::Operand& op1, const Xbyak::Operand& op2);

    void uni_vdivps(const Xbyak::Xmm& vmm_dst, const Xbyak::Operand& op1, const Xbyak::Operand& op2);

    void uni_vdivpd(const Xbyak::Xmm& vmm_dst, const Xbyak::Operand& op1, const Xbyak::Operand& op2);

    void uni_vandps(const Xbyak::Xmm& vmm_dst, const Xbyak::Xmm& vmm_src, const Xbyak::Operand &op);

    void uni_vandpd(const Xbyak::Xmm& vmm_dst, const Xbyak::Xmm& vmm_src, const Xbyak::Operand &op);

    void uni_vandnps(const Xbyak::Xmm& vmm_dst, const Xbyak::Xmm& vmm_src, const Xbyak::Operand &op);

    void uni_vorpd(const Xbyak::Xmm& vmm_dst, const Xbyak::Xmm& vmm_src, const Xbyak::Operand &op);

    void uni_vcmppd(const Xbyak::Xmm& vmm_dst, const Xbyak::Xmm& vmm_src, const Xbyak::Operand &op, const uint8_t imm);

    void uni_vmaxpd(const Xbyak::Xmm& vmm_dst, const Xbyak::Operand &op1, const Xbyak::Operand &op2);

    void uni_vminpd(const Xbyak::Xmm& vmm_dst, const Xbyak::Operand &op1, const Xbyak::Operand &op2);

    void uni_kmovd(const Xbyak::Opmask& kDst, const Xbyak::Opmask& kSrc) {
        kmovd(kDst, kSrc);
    }

    void uni_kmovd(const Xbyak::Xmm& vmm_dst, const Xbyak::Xmm& vmm_src) {
        uni_vmovups(vmm_dst, vmm_src);
    }

    void uni_kandd(const Xbyak::Opmask& kDst, const Xbyak::Opmask& kSrc1, const Xbyak::Opmask& kSrc2) {
        kandd(kDst, kSrc1, kSrc2);
    }

    void uni_kandd(const Xbyak::Xmm& kDst, const Xbyak::Xmm& kSrc1, const Xbyak::Xmm& kSrc2) {
        uni_vandps(kDst, kSrc1, kSrc2);
    }

    void uni_vpbroadcastd(const Xbyak::Xmm &vmm_dst, const Xbyak::Operand &op);

    void uni_vpbroadcastd(const Xbyak::Ymm &vmm_dst, const Xbyak::Operand &op);

    void uni_vcvtpd2dq(const Xbyak::Xmm& vmm_dst, const Xbyak::Operand &op);

    void uni_vcvtpd2ps(const Xbyak::Xmm& vmm_dst, const Xbyak::Operand &op);

    void gatherdd(
            const Xbyak::Xmm&    vmm_dst,
            const Xbyak::Reg64&  rSrcPtr,
            const Xbyak::Xmm&    vSrcShift,
            const Xbyak::Opmask& kReadMask,
            const bool useMask   = true,
            const bool zeroFill  = false);

    void gatherdd(
            const Xbyak::Xmm&   vmm_dst,
            const Xbyak::Reg64& rSrcPtr,
            const Xbyak::Xmm&   vSrcShift,
            const Xbyak::Xmm&   vReadMask,
            const bool useMask  = true,
            const bool zeroFill = false);

    void gatherdd(
            const Xbyak::Ymm&   vmm_dst,
            const Xbyak::Reg64& rSrcPtr,
            const Xbyak::Ymm&   vSrcShift,
            const Xbyak::Ymm&   vReadMask,
            const bool useMask  = true,
            const bool zeroFill = false);

    void fillRestWorkMask(
            const Xbyak::Opmask& kDstMask,
            const Xbyak::Reg64& rWorkRest);

    void fillRestWorkMask(
            const Xbyak::Xmm& ymmDstMask,
            const Xbyak::Reg64& rWorkRest,
            const uint64_t typeSize = 4);

    void fillRestWorkMask(
            const Xbyak::Ymm& ymmDstMask,
            const Xbyak::Reg64& rWorkRest,
            const uint64_t typeSize = 4);

    void load(
            const Xbyak::Xmm&     vmm_dst,
            const Xbyak::Address& adr_src,
            const Xbyak::Reg64&   rLoadNum,
            const size_t          typeSize,
            const bool zeroFill = false);

    void load(
            const Xbyak::Ymm&     vmm_dst,
            const Xbyak::Address& adr_src,
            const Xbyak::Reg64&   rLoadNum,
            const size_t          typeSize,
            const bool zeroFill = false);

    void store(
            const Xbyak::Address& dstAddr,
            const Xbyak::Xmm&     vmm_src,
            const Xbyak::Reg64&   rToStoreNum,
            const size_t          typeSize);

    void store(
            const Xbyak::Address& dstAddr,
            const Xbyak::Ymm&     vmm_src,
            const Xbyak::Reg64&   rToStoreNum,
            const size_t          typeSize);

    // Makes gather from memory under the vReadMask and writes to the memory m128.
    void memMovDD(
            const Xbyak::Reg64& rDst,
            const Xbyak::Reg64& rSrc,
            const Xbyak::Xmm&   vReadMask,
            const Xbyak::Xmm&   vSrcShift,
            const Xbyak::Reg64& rToStoreCounter,
            const bool useMask  = true,
            const bool zeroFill = false);

    // Makes gather from the memory under the vReadMask and writes to the memory m256.
    void memMovDD(
            const Xbyak::Reg64& rDst,
            const Xbyak::Reg64& rSrc,
            const Xbyak::Ymm&   vReadMask,
            const Xbyak::Ymm&   vSrcShift,
            const Xbyak::Reg64& rToStoreCounter,
            const bool useMask  = true,
            const bool zeroFill = false);

    void load_vector(
            const Xbyak::Xmm& vmm_dst,
            const Xbyak::Address &srcAdr,
            const ov::element::Type& dstPrc,
            const ov::element::Type& srcPrc);

    void load_scalar(
            const Xbyak::Xmm& vmm_dst,
            const Xbyak::Address &srcAdr,
            const ov::element::Type& dstPrc,
            const ov::element::Type& srcPrc);

    void load_with_bcst(
            const Xbyak::Xmm& vmm_dst,
            const Xbyak::Address &srcAdr,
            const ov::element::Type& dstPrc,
            const ov::element::Type& srcPrc);

    void store_vector(
            const Xbyak::Address &dstAdr,
            const Xbyak::Xmm& vmm_src,
            const ov::element::Type& dstPrc,
            const ov::element::Type& srcPrc);

    void store_scalar(
            const Xbyak::Address &dstAdr,
            const Xbyak::Xmm& vmm_src,
            const ov::element::Type& dstPrc,
            const ov::element::Type& srcPrc);

protected:
    inline bool isValidIsa(dnnl::impl::cpu::x64::cpu_isa_t isa) {
        return dnnl::impl::cpu::x64::mayiuse(isa);
    }

    RegistersPool::Ptr registersPool;

    std::shared_ptr<jit_uni_vcvtneps2bf16> vcvtneps2bf16;

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

template<typename CompileParams, typename CallArgs>
class JitKernel : public JitKernelBase {
public:
    using KernelFunc = void (*)(const CallArgs *);

    explicit JitKernel(const char* name, const CompileParams& jcp, dnnl::impl::cpu::x64::cpu_isa_t max_cpu_isa)
            : JitKernelBase{name, max_cpu_isa}, jcp{jcp}, func{nullptr} {}
    ~JitKernel() override = default;

    dnnl::impl::status_t create_kernel() override {
        const dnnl::impl::status_t code = jit_generator::create_kernel();
        if (code != dnnl::impl::status::success) {
            IE_THROW() << "Could not create kernel. Error code: " << std::to_string(code) << ". " <<
                       "Xbyak error code: " << Xbyak::ConvertErrorToString(Xbyak::GetError());
        }
        func = (decltype(func))jit_ker();
        return code;
    }

    void operator()(const CallArgs* args) const {
        assert(func);
        func(args);
    }

    void operator()(const CallArgs& args) const {
        this->operator()(&args);
    }

protected:
    CompileParams jcp;

private:
    KernelFunc func;
};

} // namespace kernel
} // namespace intel_cpu
} // namespace ov
