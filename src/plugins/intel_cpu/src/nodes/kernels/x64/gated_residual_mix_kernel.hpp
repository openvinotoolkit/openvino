// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "jit_kernel_base.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::kernel {

// GatedResidual READ/FINAL_MIX mix for one row:
//   out[d] = scale * sum_{c < branches} gate[c * data_size + d] * rbar[c * data_size + d],  d in [0, data_size).
// gate and rbar are contiguous f32 rows of branches * data_size elements; out is stored at `prc`, converting
// from f32 with a single rounding. The multiply-accumulate over branches, the scale and the conversion are one
// pass over the data.
struct jit_gated_residual_mix_compile_params {
    ov::element::Type prc;  // output element type
    size_t data_size = 0UL;
    size_t branches = 0UL;
    float scale = 1.0F;
};

struct jit_gated_residual_mix_call_args {
    const float* gate;
    const float* rbar;
    uint8_t* out;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_gated_residual_mix_kernel
    : public JitKernel<jit_gated_residual_mix_compile_params, jit_gated_residual_mix_call_args> {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_gated_residual_mix_kernel)

    static constexpr size_t vec_size = dnnl::impl::cpu::x64::cpu_isa_traits_t<isa>::vlen / sizeof(float);

    explicit jit_gated_residual_mix_kernel(const jit_gated_residual_mix_compile_params& jcp)
        : JitKernel(jit_name(), jcp, isa) {}

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                                         Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Zmm>::type;

    void generate() override;
    // out = scale * sum_c gate_c * rbar_c for `step` elements, advancing the pointers when it is a full vector.
    void mix_store(size_t step);
    // acc += gate_b * rbar_b for branches [first, first + count), relative to the given base pointers.
    void accumulate_branches(const Xbyak::Reg64& reg_gate_base,
                             const Xbyak::Reg64& reg_rbar_base,
                             size_t first,
                             size_t count,
                             int elt_num);
    void load(const Vmm& vmm_dst,
              const Xbyak::Reg64& reg_src,
              ov::element::Type src_prc,
              const int& elt_num,
              size_t offset);
    void store(const Xbyak::Reg64& reg_dst, const Vmm& vmm_src, ov::element::Type dst_prc, const int& elt_num);

    const Vmm vmm_scale = Vmm(0);
    const Vmm vmm_acc = Vmm(1);
    const Vmm vmm_gate = Vmm(2);
    const Vmm vmm_rbar = Vmm(3);
    const Xbyak::Reg64 reg_gate = r8;
    const Xbyak::Reg64 reg_rbar = r10;
    const Xbyak::Reg64 reg_out = r11;
    const Xbyak::Reg64 reg_work_amount = r12;
    const Xbyak::Reg32 reg_tmp32 = r13d;
    // Branch-group loop (used only when there are many branches).
    const Xbyak::Reg64 reg_gate_cursor = r14;
    const Xbyak::Reg64 reg_rbar_cursor = r15;
    const Xbyak::Reg64 reg_group_count = rbx;
    // Branches per group of the runtime loop. Either way at most 2 * branch_unroll - 1 branches are unrolled.
    static constexpr size_t branch_unroll = 4;

    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> emitters;
    const std::vector<size_t> pool_aux_gpr_idxs = {static_cast<size_t>(rax.getIdx()), static_cast<size_t>(r9.getIdx())};
    const std::vector<size_t> pool_aux_vmm_idxs = {7};
};

}  // namespace ov::intel_cpu::kernel
