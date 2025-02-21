// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_kernel_base.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#    include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#endif

namespace ov::intel_cpu::kernel {

struct jit_rotary_compile_params {
    ov::element::Type src_prc;
    ov::element::Type dst_prc;
    size_t rotary_ndims;
    bool interleave;
    bool mix_cos_sin;
};

struct jit_rotary_call_args {
    const void* src;
    const float* cos;
    const float* sin;
    void* dst;
};

#if defined(OPENVINO_ARCH_X86_64)

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_rotary_kernel : public JitKernel<jit_rotary_compile_params, jit_rotary_call_args> {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rotary_kernel)

    static constexpr size_t vec_size = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen / sizeof(float);

    explicit jit_rotary_kernel(const jit_rotary_compile_params& jcp) : JitKernel(jit_name(), jcp, isa) {}

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                                         Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Zmm>::type;

    void generate() override;
    void rotary_half(size_t step);
    void rotary_interleave(size_t step);
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
    const Vmm vmm_src0 = Vmm(0);
    const Vmm vmm_src1 = Vmm(1);
    const Vmm vmm_cos = Vmm(2);
    const Vmm vmm_sin = Vmm(3);
    const Vmm vmm_dst0 = Vmm(4);
    const Vmm vmm_dst1 = Vmm(5);
    const Vmm vmm_idx = Vmm(7);
    const Xbyak::Reg64 reg_src = r8;
    const Xbyak::Reg64 reg_cos = r10;
    const Xbyak::Reg64 reg_sin = r11;
    const Xbyak::Reg64 reg_dst = r12;
    const Xbyak::Reg64 reg_tmp = rdx;

    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> emitters;
    const std::vector<size_t> pool_aux_gpr_idxs = {static_cast<size_t>(rax.getIdx()), static_cast<size_t>(r9.getIdx())};
    const std::vector<size_t> pool_aux_vmm_idxs = {6};
};

#endif

}  // namespace ov::intel_cpu::kernel
