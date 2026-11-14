// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_residual_mix_kernel.hpp"

#include <xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#include "openvino/core/type/element_type.hpp"

using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov::intel_cpu::kernel {

#define GET_OFF(field) offsetof(jit_gated_residual_mix_call_args, field)

template <cpu_isa_t isa>
void jit_gated_residual_mix_kernel<isa>::generate() {
    this->preamble();

    mov(reg_gate, ptr[abi_param1 + GET_OFF(gate)]);
    mov(reg_rbar, ptr[abi_param1 + GET_OFF(rbar)]);
    mov(reg_out, ptr[abi_param1 + GET_OFF(out)]);
    mov(reg_work_amount, m_jcp.data_size);

    // The scale (1 / branches) is a compile-time constant: materialize and broadcast it once.
    uint32_t scale_bits = 0;
    std::memcpy(&scale_bits, &m_jcp.scale, sizeof(scale_bits));
    mov(reg_tmp32, scale_bits);
    uni_vmovd(Xmm(vmm_scale.getIdx()), reg_tmp32);
    uni_vbroadcastss(vmm_scale, Xmm(vmm_scale.getIdx()));

    Xbyak::Label loop_label;
    Xbyak::Label end_label;

    L(loop_label);
    {
        cmp(reg_work_amount, vec_size);
        jl(end_label, T_NEAR);

        mix_store(vec_size);

        sub(reg_work_amount, vec_size);
        jmp(loop_label, T_NEAR);
    }
    L(end_label);

    const size_t tail_size = m_jcp.data_size % vec_size;
    if (tail_size) {
        mix_store(tail_size);
    }

    this->postamble();

    for (const auto& emitter : emitters) {
        if (emitter.second) {
            emitter.second->emit_data();
        }
    }
}

template <cpu_isa_t isa>
void jit_gated_residual_mix_kernel<isa>::accumulate_branches(const Xbyak::Reg64& reg_gate_base,
                                                             const Xbyak::Reg64& reg_rbar_base,
                                                             size_t first,
                                                             size_t count,
                                                             int elt_num) {
    // Branch b lives `b * data_size` elements after the base pointers in both rows.
    const size_t branch_stride = m_jcp.data_size * sizeof(float);
    for (size_t i = 0; i < count; i++) {
        load(vmm_gate, reg_gate_base, ov::element::f32, elt_num, (first + i) * branch_stride);
        load(vmm_rbar, reg_rbar_base, ov::element::f32, elt_num, (first + i) * branch_stride);
        uni_vfmadd231ps(vmm_acc, vmm_gate, vmm_rbar);
    }
}

template <cpu_isa_t isa>
void jit_gated_residual_mix_kernel<isa>::mix_store(size_t step) {
    const bool is_tail = step < vec_size;
    const auto elt_num = static_cast<int>(step);
    const size_t branch_stride = m_jcp.data_size * sizeof(float);

    load(vmm_acc, reg_gate, ov::element::f32, elt_num, 0);
    load(vmm_rbar, reg_rbar, ov::element::f32, elt_num, 0);
    uni_vmulps(vmm_acc, vmm_acc, vmm_rbar);

    // Branches 1..C-1. Few branches are unrolled at generation time; many run in a loop over groups of
    // branch_unroll branches, so the code size stays bounded however large C is. The remainder is unrolled.
    const size_t rest = m_jcp.branches - 1;
    const size_t groups = rest / branch_unroll;
    if (groups >= 2) {
        lea(reg_gate_cursor, ptr[reg_gate + branch_stride]);
        lea(reg_rbar_cursor, ptr[reg_rbar + branch_stride]);
        mov(reg_group_count, groups);
        Xbyak::Label group_loop;
        L(group_loop);
        {
            accumulate_branches(reg_gate_cursor, reg_rbar_cursor, 0, branch_unroll, elt_num);
            add(reg_gate_cursor, branch_unroll * branch_stride);
            add(reg_rbar_cursor, branch_unroll * branch_stride);
            dec(reg_group_count);
            jnz(group_loop, T_NEAR);
        }
        accumulate_branches(reg_gate_cursor, reg_rbar_cursor, 0, rest % branch_unroll, elt_num);
    } else {
        accumulate_branches(reg_gate, reg_rbar, 1, rest, elt_num);
    }

    uni_vmulps(vmm_acc, vmm_acc, vmm_scale);
    store(reg_out, vmm_acc, m_jcp.prc, elt_num);

    if (!is_tail) {
        add(reg_gate, sizeof(float) * step);
        add(reg_rbar, sizeof(float) * step);
        add(reg_out, m_jcp.prc.size() * step);
    }
}

template <cpu_isa_t isa>
void jit_gated_residual_mix_kernel<isa>::load(const Vmm& vmm_dst,
                                              const Xbyak::Reg64& reg_src,
                                              ov::element::Type src_prc,
                                              const int& elt_num,
                                              size_t offset) {
    const auto seed = load_emitter_params(src_prc, ov::element::f32, elt_num, false, "float_min").hash();
    if (!emitters[seed]) {
        emitters[seed] = std::make_unique<jit_load_emitter>(this,
                                                            isa,
                                                            src_prc,
                                                            ov::element::f32,
                                                            elt_num,
                                                            ov::element::f32,
                                                            false,
                                                            "float_min");
    }
    emitters[seed]->emit_code({static_cast<size_t>(reg_src.getIdx()), offset},
                              {static_cast<size_t>(vmm_dst.getIdx())},
                              pool_aux_vmm_idxs,
                              pool_aux_gpr_idxs);
}

template <cpu_isa_t isa>
void jit_gated_residual_mix_kernel<isa>::store(const Xbyak::Reg64& reg_dst,
                                               const Vmm& vmm_src,
                                               ov::element::Type dst_prc,
                                               const int& elt_num) {
    const auto seed = store_emitter_params(ov::element::f32, dst_prc, elt_num).hash();
    if (!emitters[seed]) {
        emitters[seed] = std::make_unique<jit_store_emitter>(this, isa, ov::element::f32, dst_prc, elt_num);
    }
    emitters[seed]->emit_code({static_cast<size_t>(vmm_src.getIdx())},
                              {static_cast<size_t>(reg_dst.getIdx()), 0},
                              pool_aux_vmm_idxs,
                              pool_aux_gpr_idxs);
}

template struct jit_gated_residual_mix_kernel<cpu_isa_t::avx512_core>;
template struct jit_gated_residual_mix_kernel<cpu_isa_t::avx2>;

}  // namespace ov::intel_cpu::kernel
