// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#include "jit_kernel_base.hpp"

namespace ov::intel_cpu::kernel {

struct jit_rms_compile_params {
    ov::element::Type src_prc;
    ov::element::Type dst_prc;
    size_t data_size;
    float eps;
    size_t scale_size;
};

struct jit_rms_call_args {
    const uint8_t* src;
    const float* scale;
    uint8_t* dst;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_rms_kernel : public JitKernel<jit_rms_compile_params, jit_rms_call_args> {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rms_kernel)

    static constexpr size_t vec_size = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen / sizeof(float);

    explicit jit_rms_kernel(const jit_rms_compile_params& jcp) : JitKernel(jit_name(), jcp, isa) {}

private:
    using Xmm = Xbyak::Xmm;
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                                         Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Zmm>::type;

    void generate() override;
    void load(const Vmm& vmm_dst,
              const Xbyak::Reg64& reg_src,
              ov::element::Type src_prc,
              const int& elt_num,
              bool fill,
              size_t offset = 0);
    void store(const Xbyak::Reg64& reg_dst,
               const Vmm& vmm_src,
               ov::element::Type dst_prc,
               const int& elt_num,
               size_t offset = 0);

    // from onednn
    void reduce_zmm_to_ymm(const Xmm& acc, const Xmm& tmp);
    void reduce_ymm_to_xmm(const Xmm& acc, const Xmm& tmp);
    void reduce_xmm_to_scalar(const Xmm& acc,
                              const Xmm& tmp,
                              const std::size_t number_of_values_to_reduce = number_of_f32_in_xmm_);
    void reduce_ymm_to_scalar(const Xbyak::Xmm& acc,
                              const Xbyak::Xmm& tmp1,
                              const Xbyak::Xmm& tmp2,
                              const std::size_t number_of_values_to_reduce = number_of_f32_in_ymm_);
    void reduce_vmm_to_scalar(const Xbyak::Xmm& acc,
                              const Xbyak::Xmm& tmp1,
                              const Xbyak::Xmm& tmp2,
                              const Xbyak::Xmm& tmp3,
                              const std::size_t number_of_values_to_reduce = number_of_f32_in_zmm_);
    static constexpr std::size_t number_of_f32_in_xmm_ = 4;
    static constexpr std::size_t number_of_f32_in_ymm_ = 8;
    static constexpr std::size_t number_of_f32_in_zmm_ = 16;

    const Vmm vmm_src = Vmm(0);
    const Vmm vmm_sum0 = Vmm(2);
    const Vmm vmm_rsqrt = Vmm(2);
    const Xmm xmm_rsqrt = Xmm(2);
    const Vmm vmm_sum1 = Vmm(3);
    const Vmm vmm_tmp = Vmm(3);
    const Xmm xmm_tmp = Xmm(3);
    const Vmm vmm_sum2 = Vmm(4);
    const Vmm vmm_sum3 = Vmm(5);
    const Vmm vmm_dst = Vmm(6);
    const Xbyak::Reg64 reg_src = r8;
    const Xbyak::Reg64 reg_src_org = r13;
    const Xbyak::Reg64 reg_scale = r10;
    const Xbyak::Reg64 reg_size = r11;
    const Xbyak::Reg64 reg_dst = r12;
    const Xbyak::Reg64 reg_tmp = rdx;

    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> emitters;
    const std::vector<size_t> pool_aux_gpr_idxs = {static_cast<size_t>(rax.getIdx()), static_cast<size_t>(r9.getIdx())};
    const std::vector<size_t> pool_aux_vmm_idxs = {7};
};

}  // namespace ov::intel_cpu::kernel
