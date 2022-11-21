// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <cpu/x64/jit_generator.hpp>

namespace ov {
namespace intel_cpu {

using namespace dnnl::impl;
using namespace dnnl::impl::cpu;

class jit_base : public x64::jit_generator {
public:
    static constexpr size_t xmm_len = 16;
    static constexpr size_t ymm_len = 32;
    static constexpr size_t zmm_len = 64;

    explicit jit_base(const char* name, x64::cpu_isa_t max_cpu_isa)
        : x64::jit_generator{name, nullptr, dnnl::impl::cpu::x64::MAX_CODE_SIZE, true, max_cpu_isa}
        , max_cpu_isa_{ max_cpu_isa } {}
    virtual ~jit_base() = default;

    bool is_valid_isa(const x64::cpu_isa_t isa) const {
        return x64::is_subset(isa, max_cpu_isa_) && x64::mayiuse(isa);
    }

    using x64::jit_generator::push;
    using x64::jit_generator::pop;

    void push(const Xbyak::Xmm& xmm);
    void pop(const Xbyak::Xmm& xmm);

    void uni_vaddps(const Xbyak::Xmm& x, const Xbyak::Xmm& op1, const Xbyak::Operand& op2);

    void uni_vaddps(const Xbyak::Ymm& x, const Xbyak::Ymm& op1, const Xbyak::Operand& op2) {
        vaddps(x, op1, op2);
    }

    void uni_vsubps(const Xbyak::Xmm& x, const Xbyak::Xmm& op1, const Xbyak::Operand& op2);

    void uni_vsubps(const Xbyak::Ymm& x, const Xbyak::Ymm& op1, const Xbyak::Operand& op2) {
        vsubps(x, op1, op2);
    }

    void uni_vcmpps(const Xbyak::Xmm& x, const Xbyak::Xmm& op1, const Xbyak::Operand& op2, const int cmp_predicate);

    void uni_vcmpps(const Xbyak::Ymm& x1, const Xbyak::Ymm& x2, const Xbyak::Operand& op, const int cmp_predicate) {
        vcmpps(x1, x2, op, cmp_predicate);
    }

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

    void uni_vandps(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrs, const Xbyak::Operand& op);

    void uni_vandnps(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrs, const Xbyak::Operand& op);

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

    void uni_vpbroadcastd(const Xbyak::Xmm& x, const Xbyak::Operand& op);

    void uni_vpbroadcastd(const Xbyak::Ymm& x, const Xbyak::Operand& op);

    void load(const Xbyak::Xmm& vDst,
              const Xbyak::Address& srcAddr,
              const Xbyak::Reg64& rLoadNum,
              const size_t typeSize,
              const bool zeroFill = false);

    void load(const Xbyak::Ymm& vDst,
              const Xbyak::Address& srcAddr,
              const Xbyak::Reg64& rLoadNum,
              const size_t typeSize,
              const bool zeroFill = false);

    void store(const Xbyak::Address& dstAddr,
               const Xbyak::Xmm& vSrc,
               const Xbyak::Reg64& rToStoreNum,
               const size_t typeSize);

    void store(const Xbyak::Address& dstAddr,
               const Xbyak::Ymm& vSrc,
               const Xbyak::Reg64& rToStoreNum,
               const size_t typeSize);

    static constexpr unsigned VCMPPS_LE = 0x02;
    static constexpr unsigned VCMPPS_GT = 0x0e;

protected:
    x64::cpu_isa_t max_cpu_isa_;
};

} // namespace intel_cpu
} // namespace ov
