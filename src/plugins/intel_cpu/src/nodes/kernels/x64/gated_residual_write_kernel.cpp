// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gated_residual_write_kernel.hpp"

#include <xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>

#include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#include "openvino/core/type/element_type.hpp"

using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov::intel_cpu::kernel {

#define GET_OFF(field) offsetof(jit_gated_residual_write_call_args, field)

template <cpu_isa_t isa>
void jit_gated_residual_write_kernel<isa>::generate() {
    this->preamble();

    mov(reg_residual, ptr[abi_param1 + GET_OFF(residual)]);
    mov(reg_y, ptr[abi_param1 + GET_OFF(y)]);
    mov(reg_gate, ptr[abi_param1 + GET_OFF(gate)]);
    mov(reg_out, ptr[abi_param1 + GET_OFF(out)]);
    mov(reg_work_amount, m_jcp.data_size);

    // The gate is one element per call: load it (converting to f32) and broadcast it once.
    load(vmm_gate, reg_gate, m_jcp.prc, 1);
    uni_vbroadcastss(vmm_gate, Xmm(vmm_gate.getIdx()));

    Xbyak::Label loop_label;
    Xbyak::Label end_label;

    L(loop_label);
    {
        cmp(reg_work_amount, vec_size);
        jl(end_label, T_NEAR);

        load_fma_store(vec_size);

        sub(reg_work_amount, vec_size);
        jmp(loop_label, T_NEAR);
    }
    L(end_label);

    const size_t tail_size = m_jcp.data_size % vec_size;
    if (tail_size) {
        load_fma_store(tail_size);
    }

    this->postamble();

    for (const auto& emitter : emitters) {
        if (emitter.second) {
            emitter.second->emit_data();
        }
    }
}

template <cpu_isa_t isa>
void jit_gated_residual_write_kernel<isa>::load_fma_store(size_t step) {
    const bool is_tail = step < vec_size;
    load(vmm_residual, reg_residual, m_jcp.prc, static_cast<int>(step));
    load(vmm_y, reg_y, m_jcp.prc, static_cast<int>(step));
    // residual += y * gate
    uni_vfmadd231ps(vmm_residual, vmm_y, vmm_gate);
    store(reg_out, vmm_residual, m_jcp.prc, static_cast<int>(step));

    if (!is_tail) {
        add(reg_residual, m_jcp.prc.size() * step);
        add(reg_y, m_jcp.prc.size() * step);
        add(reg_out, m_jcp.prc.size() * step);
    }
}

template <cpu_isa_t isa>
void jit_gated_residual_write_kernel<isa>::load(const Vmm& vmm_dst,
                                                const Xbyak::Reg64& reg_src,
                                                ov::element::Type src_prc,
                                                const int& elt_num) {
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
    emitters[seed]->emit_code({static_cast<size_t>(reg_src.getIdx()), 0},
                              {static_cast<size_t>(vmm_dst.getIdx())},
                              pool_aux_vmm_idxs,
                              pool_aux_gpr_idxs);
}

template <cpu_isa_t isa>
void jit_gated_residual_write_kernel<isa>::store(const Xbyak::Reg64& reg_dst,
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

template struct jit_gated_residual_write_kernel<cpu_isa_t::avx512_core>;
template struct jit_gated_residual_write_kernel<cpu_isa_t::avx2>;

}  // namespace ov::intel_cpu::kernel
