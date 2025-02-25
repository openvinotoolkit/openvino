// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_quantization_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "emitters/plugin/x64/jit_bf16_emitters.hpp"
#include "emitters/plugin/x64/jit_emitter.hpp"
#include "nodes/executors/eltwise.hpp"
#include "nodes/kernels/jit_eltwise_common.hpp"
#include "onednn/dnnl.h"
#include "utils/cpu_utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::x64 {

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_uni_eltwise_generic : public jit_uni_eltwise_kernel, public dnnl::impl::cpu::x64::jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_eltwise_generic)

    jit_uni_eltwise_generic(const jit_eltwise_params& jep,
                            const std::vector<EltwiseData>& eltwise_data,
                            const std::vector<ov::intel_cpu::Type>& ops_list,
                            const dnnl::post_ops& post_ops);

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override;

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                                         Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Zmm>::type;

    inline Xbyak::Reg64 get_src_reg(int idx) {
        return Xbyak::Reg64(r8.getIdx() + idx);
    }

    inline Vmm get_vmm_reg(int idx) {
        return Vmm(1 + idx);
    }

    inline Vmm get_aux_vmm(int idx) {
        return Vmm(10 + idx);
    }

    inline Xbyak::Xmm get_xmm_reg(int idx) {
        return Xbyak::Xmm(get_vmm_reg(idx).getIdx());
    }

    Xbyak::Reg64 reg_post_op_ptrs = rax;
    Xbyak::Reg64 start_to_offsets = reg_post_op_ptrs;  // rax
    Xbyak::Reg64 reg_dst = rbx;
    Xbyak::Reg64 reg_work_amount = rdx;

    static constexpr auto abi_param_regs = dnnl::impl::cpu::x64::abi_param_regs;
    static constexpr auto abi_not_param_reg = dnnl::impl::cpu::x64::abi_not_param_reg;
    Xbyak::Reg64 reg_oc_off = abi_not_param1;
    Xbyak::Reg64 reg_const_params = abi_param1;
    Xbyak::Reg64 reg_indexes = abi_param2;  // reg_d_bias

    Xbyak::Reg8 reg_tmp_8 = Xbyak::Reg8(r15.getIdx());
    Xbyak::Reg16 reg_tmp_16 = Xbyak::Reg16(r15.getIdx());
    Xbyak::Reg32 reg_tmp_32 = Xbyak::Reg32(r15.getIdx());
    Xbyak::Reg64 reg_tmp_64 = Xbyak::Reg64(r15.getIdx());

    Xbyak::Reg64 reg_d_weights = rbp;
    Xbyak::Reg64 reg_d_bias = rsi;

    Vmm vmm_dst = Vmm(9);
    Xbyak::Xmm xmm_dst = Xbyak::Xmm(9);

    Vmm vmm_d_weights = Vmm(12);
    Vmm vmm_d_bias = Vmm(13);
    Vmm vmm_zero = Vmm(15);

    std::shared_ptr<jit_uni_vcvtneps2bf16> uni_vcvtneps2bf16;

    std::shared_ptr<jit_emitter> eltwise_emitter = nullptr;
    std::vector<std::shared_ptr<jit_emitter>> post_op_emitters = {};

    std::vector<std::shared_ptr<dnnl::impl::cpu::x64::jit_uni_quantization_injector_f32<isa>>> quantization_injectors =
        {};

    const std::vector<EltwiseData>& eltwise_data_;
    const std::vector<ov::intel_cpu::Type>& ops_list_;
    const dnnl::post_ops& post_ops_;

    std::shared_ptr<jit_emitter> create_eltwise_emitter(const EltwiseData& data, ov::element::Type exec_prec);

    void compute_eltwise_op();
    void apply_post_ops(bool is_scalar, int offset = 0);

    void load_vector(Vmm vmm_src,
                     const Xbyak::Address& op,
                     ov::element::Type src_prc,
                     ov::element::Type dst_prc,
                     bool broadcast);
    void load_scalar(Xbyak::Xmm xmm_src,
                     const Xbyak::Address& op,
                     ov::element::Type src_prc,
                     ov::element::Type dst_prc);

    void store_vector(const Xbyak::Address& op, Vmm vmm_dst, ov::element::Type src_prc, ov::element::Type dst_prc);
    void store_scalar(const Xbyak::Address& op,
                      Xbyak::Xmm xmm_dst,
                      ov::element::Type src_prc,
                      ov::element::Type dst_prc);
};

}  // namespace ov::intel_cpu::x64
