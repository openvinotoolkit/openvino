// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa_score_kernel.hpp"

#include <xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <memory>

#include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#include "openvino/core/type/element_type.hpp"

using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov::intel_cpu::kernel {

#define GET_OFF(field) offsetof(jit_qsa_score_call_args, field)

template <cpu_isa_t isa>
void jit_qsa_score_kernel<isa>::load(const Vmm& vmm_dst,
                                     const Xbyak::Reg64& reg_src,
                                     ov::element::Type src_prc,
                                     size_t offset) {
    const auto seed = load_emitter_params(src_prc, ov::element::f32, vec_size, false, "float_min").hash();
    if (!emitters[seed]) {
        emitters[seed] = std::make_unique<jit_load_emitter>(this,
                                                            isa,
                                                            src_prc,
                                                            ov::element::f32,
                                                            vec_size,
                                                            ov::element::f32,
                                                            false,
                                                            "float_min");
    }
    emitters[seed]->emit_code({static_cast<size_t>(reg_src.getIdx()), offset},
                              {static_cast<size_t>(vmm_dst.getIdx())},
                              pool_aux_vmm_idxs,
                              pool_aux_gpr_idxs);
}

// The sum of the lanes of `acc` into its lowest lane (`tmp` is clobbered).
template <cpu_isa_t isa>
void jit_qsa_score_kernel<isa>::reduce_to_scalar(const Vmm& acc, const Vmm& tmp) {
    if constexpr (isa == avx512_core) {
        vextractf64x4(Ymm(tmp.getIdx()), Zmm(acc.getIdx()), 1);
        vaddps(Ymm(acc.getIdx()), Ymm(acc.getIdx()), Ymm(tmp.getIdx()));
    }
    vextractf128(Xmm(tmp.getIdx()), Ymm(acc.getIdx()), 1);
    vaddps(Xmm(acc.getIdx()), Xmm(acc.getIdx()), Xmm(tmp.getIdx()));
    vhaddps(Xmm(acc.getIdx()), Xmm(acc.getIdx()), Xmm(acc.getIdx()));
    vhaddps(Xmm(acc.getIdx()), Xmm(acc.getIdx()), Xmm(acc.getIdx()));
}

// The code for `rows` rows starting at reg_rows: scores[0, rows) += ...
template <cpu_isa_t isa>
void jit_qsa_score_kernel<isa>::emit_rows(size_t rows) {
    const size_t dim = m_jcp.dim;
    const size_t chunks = dim / vec_size;
    const size_t parts = chunks > 1 ? 2 : 1;
    const bool convert = m_jcp.row_prc != ov::element::f32;
    const size_t row_bytes = dim * m_jcp.row_prc.size();

    // the rows as f32: in place, or converted once into the stage
    if (convert) {
        for (size_t r = 0; r < rows; r++) {
            for (size_t c = 0; c < chunks; c++) {
                load(vmm_q, reg_rows, m_jcp.row_prc, r * row_bytes + c * vec_size * m_jcp.row_prc.size());
                uni_vmovups(ptr[reg_stage + (r * dim + c * vec_size) * sizeof(float)], vmm_q);
            }
        }
        mov(reg_rowbase, reg_stage);
    } else {
        mov(reg_rowbase, reg_rows);
    }

    for (size_t r = 0; r < rows; r++) {
        uni_vpxor(row_sum(r), row_sum(r), row_sum(r));
    }
    mov(reg_qh, reg_q);
    mov(reg_heads, m_jcp.heads);

    Xbyak::Label head_loop;
    L(head_loop);
    {
        for (size_t r = 0; r < rows; r++) {
            for (size_t part = 0; part < parts; part++) {
                uni_vpxor(acc(r, part), acc(r, part), acc(r, part));
            }
        }
        for (size_t c = 0; c < chunks; c++) {
            uni_vmovups(vmm_q, ptr[reg_qh + c * vec_size * sizeof(float)]);
            for (size_t r = 0; r < rows; r++) {
                vfmadd231ps(acc(r, c % parts), vmm_q, ptr[reg_rowbase + (r * dim + c * vec_size) * sizeof(float)]);
            }
        }
        for (size_t r = 0; r < rows; r++) {
            if (parts == 2) {
                vaddps(acc(r, 0), acc(r, 0), acc(r, 1));
            }
            reduce_to_scalar(acc(r, 0), vmm_tmp);
            // max(0, q . row), summed over the heads
            vmaxss(Xmm(acc(r, 0).getIdx()), Xmm(acc(r, 0).getIdx()), xmm_zero);
            vaddss(row_sum(r), row_sum(r), Xmm(acc(r, 0).getIdx()));
        }
        add(reg_qh, dim * sizeof(float));
        dec(reg_heads);
        jnz(head_loop, T_NEAR);
    }

    for (size_t r = 0; r < rows; r++) {
        vmovss(xmm_tmp, ptr[reg_scores + r * sizeof(float)]);
        vaddss(xmm_tmp, xmm_tmp, row_sum(r));
        vmovss(ptr[reg_scores + r * sizeof(float)], xmm_tmp);
    }
}

template <cpu_isa_t isa>
void jit_qsa_score_kernel<isa>::generate() {
    this->preamble();
    mov(reg_q, ptr[abi_param1 + GET_OFF(q)]);
    mov(reg_rows, ptr[abi_param1 + GET_OFF(rows)]);
    mov(reg_scores, ptr[abi_param1 + GET_OFF(scores)]);
    mov(reg_stage, ptr[abi_param1 + GET_OFF(stage)]);
    mov(reg_n, ptr[abi_param1 + GET_OFF(n)]);
    uni_vpxor(xmm_zero, xmm_zero, xmm_zero);

    const size_t row_bytes = m_jcp.dim * m_jcp.row_prc.size();
    Xbyak::Label rows_loop;
    Xbyak::Label single_loop;
    Xbyak::Label done;

    L(rows_loop);
    cmp(reg_n, kRows);
    jl(single_loop, T_NEAR);
    emit_rows(kRows);
    add(reg_rows, kRows * row_bytes);
    add(reg_scores, kRows * sizeof(float));
    sub(reg_n, kRows);
    jmp(rows_loop, T_NEAR);

    L(single_loop);
    cmp(reg_n, 0);
    jle(done, T_NEAR);
    emit_rows(1);
    add(reg_rows, row_bytes);
    add(reg_scores, sizeof(float));
    dec(reg_n);
    jmp(single_loop, T_NEAR);

    L(done);
    this->postamble();
    for (const auto& emitter : emitters) {
        if (emitter.second) {
            emitter.second->emit_data();
        }
    }
}

template struct jit_qsa_score_kernel<avx2>;
template struct jit_qsa_score_kernel<avx512_core>;

}  // namespace ov::intel_cpu::kernel
