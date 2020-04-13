/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
* Copyright 2018 YANDEX LLC
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_JIT_UNI_POOL_KERNEL_HPP
#define CPU_JIT_UNI_POOL_KERNEL_HPP

#include <cfloat>

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "type_helpers.hpp"

#include "jit_primitive_conf.hpp"
#include "jit_avx512_core_bf16cvt.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

template <cpu_isa_t isa>
struct jit_uni_pool_kernel: public jit_generator {
    jit_uni_pool_kernel(jit_pool_conf_t ajpp):
        jpp(ajpp), bf16_emu_(nullptr)
    {
        if (jpp.is_bf16 && !isa_has_bf16(jpp.isa))
            bf16_emu_ = new bf16_emulation_t(this,
                    bf16_emu_reserv_1, bf16_emu_reserv_2,
                    bf16_emu_reserv_3, bf16_emu_reserv_4,
                    bf16_emu_reserv_5, bf16_emu_reserv_5);

        this->generate();
        jit_ker = (decltype(jit_ker))this->getCode();
    }

    ~jit_uni_pool_kernel() { delete bf16_emu_; }

    jit_pool_conf_t jpp;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_pool_kernel)

    void operator()(jit_pool_call_s *arg) { jit_ker(arg); }
    static status_t init_conf(jit_pool_conf_t &jbp,
            const pooling_desc_t &pd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &dst_d);

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm, isa == avx,
                                             Ymm, Zmm>::type;
    Xmm xreg(int idx) { return Xmm((isa == avx512_common ? 31 : 15) - idx); }
    Ymm yreg(int idx) { return Ymm(xreg(idx).getIdx()); }
    Zmm zreg(int idx) { return Zmm(xreg(idx).getIdx()); }
    Vmm vreg(int idx) { return Vmm(xreg(idx).getIdx()); }

    const AddressFrame &vmmword = (isa == sse42) ? xword :
                                  (isa == avx) ? yword : zword;

    Xmm vmm_mask = Xmm(0);
    Ymm ymm_tmp_1 = Ymm(0);
    Vmm vmm_tmp_1 = Vmm(0);

    Xmm xmm_ker_area_h = Xmm(2);
    Xmm xmm_one = Xmm(2);
    Xmm xmm_tmp = Xmm(3);

    Vmm vmm_ker_area_h = Vmm(2);
    Vmm vmm_one = Vmm(2);
    Vmm vmm_tmp = Vmm(3);
    Ymm ymm_tmp = Ymm(3);

    Vmm vmm_k_offset = Vmm(1);

    inline Vmm vmm_idx() {
        if (!jpp.is_backward) {
            return (jpp.is_training) ? Vmm(4) : Vmm(1);
        } else
            return Vmm(4);
    }

    Zmm bf16_emu_reserv_1 = Zmm(5);
    Zmm bf16_emu_reserv_2 = Zmm(6);
    Zmm bf16_emu_reserv_3 = Zmm(7);
    Reg64 bf16_emu_reserv_4 = r11;
    Zmm bf16_emu_reserv_5 = Zmm(8);

    Opmask k_index_mask = Opmask(6);
    Opmask k_store_mask = Opmask(7);
    Opmask k_mask_cvt = Opmask(5);

    // Here be some (tame) dragons. This kernel does not follow the regular
    // OS-agnostic ABI pattern because when isa is sse42 it uses maskmovdqu
    // instruction which has its destination hardcoded in rdi. Therefore:
    // - all registers are hardcoded
    // - on Windows rdi and rcx are swapped to mimic the Unix x86_64 ABI
    //
    // While this is only required by the backward pass, the quirk above
    // is applied to the forward pass as well to keep things simpler.

    using reg64_t = const Xbyak::Reg64;
    reg64_t reg_param      = rdi; // Always mimic the Unix ABI
    reg64_t reg_input      = r8;
    reg64_t aux_reg_input  = r9;
    reg64_t reg_index      = r10;
    reg64_t reg_output     = r12;
    reg64_t reg_kd_pad_shift = r13;
    reg64_t dst_ptr        = rdi; // Must be rdi due to maskmovdqu

    reg64_t kj      = r14;
    reg64_t oi_iter = r15;
    reg64_t reg_kh  = rax;
    reg64_t reg_k_shift  = rbx;
    reg64_t tmp_gpr = rcx; // Must be rcx because rdi is used above
    reg64_t reg_ker_area_h = rdx;

    reg64_t zero_size = r15;
    reg64_t ki = r12;
    reg64_t aux_reg_input_d = r8;

    Xbyak::Reg32 reg_shuf_mask = esi;

    int prev_kw;
    void (*jit_ker)(jit_pool_call_s *);

    void maybe_recalculate_divisor(int jj, int ur_w, int pad_l, int pad_r, int pad_r_logic);
    void avg_step(int ur_w, int pad_l, int pad_r, int pad_r_logic);
    void max_step_fwd(int ur_w, int pad_l, int pad_r);
    void max_step_bwd(int ur_w, int pad_l, int pad_r);

    void maybe_zero_diff_src();

    void load(int idx, reg64_t reg_ptr, int offset) {
        if (jpp.is_bf16) {
            /*TODO: maybe use vpmovzxwd + vpslld,
             * in order to free up vmm_idx() register */
            vmovups(yreg(idx), ptr[reg_ptr + offset]);
            vpermw(vreg(idx) | k_mask_cvt | T_z, vmm_idx(), vreg(idx));
        } else {
            uni_vmovups(vreg(idx), ptr[reg_ptr + offset]);
        }
    };

    void step(int ur_w, int pad_l, int pad_r, int pad_r_logic) {
        if (jpp.alg == alg_kind::pooling_max) {
            if(jpp.is_backward)
                max_step_bwd(ur_w, pad_l, pad_r);
            else
                max_step_fwd(ur_w, pad_l, pad_r);
        }
        else
            avg_step(ur_w, pad_l, pad_r, pad_r_logic);
    }

    void step_high_half(int ur_w, int pad_l, int pad_r, int pad_r_logic) {
        add(reg_input, sizeof(float) * 4);
        add(reg_output, sizeof(float) * 4);
        if (jpp.alg == alg_kind::pooling_max &&
                (jpp.is_training || jpp.is_backward))
            add(reg_index, types::data_type_size(jpp.ind_dt) * 4);

        step(ur_w, pad_l, pad_r, pad_r_logic);
    }

    void generate();

    void avx_vpadd1(const Ymm& y0, const Xmm& x1, const Xmm& xtmp) {
        assert(y0.getIdx() != x1.getIdx());
        vextractf128(xtmp, y0, 0);
        vpaddd(xtmp, xtmp, x1);
        vinsertf128(y0, y0, xtmp, 0);
        vextractf128(xtmp, y0, 1);
        vpaddd(xtmp, xtmp, x1);
        vinsertf128(y0, y0, xtmp, 1);
    }

    void avx_vpadd1(const Xmm& x0, const Xmm& x1, const Xmm&) {
        assert(false /*function should not be used*/);
        paddd(x0, x1);
    }

    void avx_pmovzxbd(const Ymm& y0, const Xmm& x1, const Xmm& xtmp) {
        Xmm x0(y0.getIdx());
        pshufd(xmm_tmp, x1, 1);
        pmovzxbd(x0, x1);
        pmovzxbd(xmm_tmp, xmm_tmp);
        vinsertf128(y0, y0, xmm_tmp, 1);
    }

    void avx_pmovzxbd(const Xmm& x0, const Xmm& x1, const Xmm&) {
        assert(false /*function should not be used*/);
        pmovzxbd(x0, x1);
    }

    void avx_pcmpeqd(const Ymm& y0, const Ymm& y1, const Ymm& y2, const Xmm& xtmp) {
        assert(y0.getIdx() != y1.getIdx());
        assert(y0.getIdx() != y2.getIdx());
        Xmm x0(y0.getIdx());
        Xmm x2(y2.getIdx());
        vextractf128(x0, y1, 1);
        vextractf128(xtmp, y2, 1);
        pcmpeqd(xtmp, x0);
        vextractf128(x0, y1, 0);
        pcmpeqd(x0, x2);
        vinsertf128(y0, y0, xtmp, 1);
    }

    void avx_pcmpeqd(const Xmm& x0, const Xmm& x1, const Xmm&, const Xmm&) {
        assert(false /*function should not be used*/);
        pcmpeqd(x0, x1);
    }

    bf16_emulation_t *bf16_emu_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
