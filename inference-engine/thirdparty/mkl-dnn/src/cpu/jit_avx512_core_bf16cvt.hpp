/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef _JIT_AVX512_CORE_BF16CVT_HPP
#define _JIT_AVX512_CORE_BF16CVT_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "mkldnn_debug.h"
#include "nstl.hpp"
#include "type_helpers.hpp"

#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace bf16_cvt_utils {
struct jit_call_t {
    void *inp;
    void *out;
    void *add;
    size_t size;
};
}

#define GET_OFF(field) offsetof(bf16_cvt_utils::jit_call_t, field)

struct bf16_emulation_t {
    using opmask_t = const Xbyak::Opmask;
    using Zmm_t = const Xbyak::Zmm;
    using Ymm_t = const Xbyak::Ymm;
    using reg64_t = const Xbyak::Reg64;

    bf16_emulation_t(jit_generator *host, Zmm_t one, Zmm_t even,
            Zmm_t selector, reg64_t scratch, Zmm_t tr0, Zmm_t tr1)
        : one_(one)
        , even_(even)
        , selector_(selector)
        , tr0_(tr0)
        , tr1_(tr1)
        , scratch_(scratch)
        , host_(host) {}

    void r_vdpbf16ps(Zmm_t &acc, Zmm_t wei, Zmm_t inp) {
        host_->vpsrad(tr0_, wei, 16);
        host_->vpslld(tr0_, tr0_, 16);

        host_->vpsrad(tr1_, inp, 16);
        host_->vpslld(tr1_, tr1_, 16);

        host_->vfmadd231ps(acc, tr1_, tr0_);

        host_->vpslld(tr0_, wei, 16);
        host_->vpslld(tr1_, inp, 16);

        host_->vfmadd231ps(acc, tr1_, tr0_);
    }

    void r_vcvtneps2bf16(Ymm_t &out, Zmm_t in) {
        host_->vpsrld(tr0_, in, 16);
        host_->vpandd(tr0_, tr0_, one_);

        host_->vpaddd(tr0_, even_, tr0_);

        host_->vpaddd(tr0_, in, tr0_);
        host_->vfixupimmps(tr0_, in, selector_, 0);

        host_->vpsrad(tr0_, tr0_, 16);
        host_->vpmovdw(out, tr0_);
    }

    void init_vcvtneps2bf16() {
        const int selector_int32 =
            /* qnan input to qnan output (presenrving input bits 0..21) */
            encode_fixup_selector(fixup_input_code_snan_, fixup_output_code_qnan_input_) |
            /* snan input to qnan output (presenrving input bits 0..21) */
            encode_fixup_selector(fixup_input_code_qnan_, fixup_output_code_qnan_input_) |
            /* neg inf input copied to output */
            encode_fixup_selector(fixup_input_code_ninf_, fixup_output_code_copy_input_) |
            /* pos inf input copied to output */
            encode_fixup_selector(fixup_input_code_pinf_, fixup_output_code_copy_input_);

        host_->xor_(scratch_, scratch_);
        host_->mov(scratch_.cvt32(), 0x1);
        host_->vpbroadcastd(one_, scratch_.cvt32());

        host_->xor_(scratch_, scratch_);
        host_->mov(scratch_.cvt32(), 0x7fff);
        host_->vpbroadcastd(even_, scratch_.cvt32());

        host_->xor_(scratch_, scratch_);
        host_->mov(scratch_.cvt32(), selector_int32);
        host_->vpbroadcastd(selector_, scratch_.cvt32());
    }

private:
    Zmm_t one_;
    Zmm_t even_;
    Zmm_t selector_;
    Zmm_t tr0_;
    Zmm_t tr1_;
    reg64_t scratch_;
    jit_generator *const host_;

    inline int encode_fixup_selector(int input, int output) {
        return ((output) << (4 * (input)));
    }

    enum {
        fixup_input_code_qnan_ = 0,
        fixup_input_code_snan_ = 1,
        fixup_input_code_ninf_ = 4,
        fixup_input_code_pinf_ = 5,
        fixup_output_code_copy_input_ = 1,
        fixup_output_code_qnan_input_ = 2,
    };

};

struct jit_avx512_core_cvt_ps_to_bf16_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_cvt_ps_to_bf16)

    jit_avx512_core_cvt_ps_to_bf16_t(void) : simd_w_(16), is_dynamic_size_(true) {
        bf16_emu_ = new bf16_emulation_t(this, one, even,
                selector, scratch, fp32_tmp, fp32_tmp);

        generate();
        jit_ker = (void (*)(bf16_cvt_utils::jit_call_t *))getCode();
    }

    jit_avx512_core_cvt_ps_to_bf16_t(size_t size)
        : size_(size), simd_w_(16), is_dynamic_size_(false) {
        tail_mask_ = (1 << (size % simd_w_)) - 1;

        bf16_emu_ = new bf16_emulation_t(this, one, even,
                selector, scratch, fp32_tmp, fp32_tmp);

        generate();
        jit_ker = (void (*)(bf16_cvt_utils::jit_call_t *))getCode();
    }

    ~jit_avx512_core_cvt_ps_to_bf16_t() { delete bf16_emu_; }

    void generate() {
        preamble();

        auto cvt = [&](size_t idx, Xbyak::Opmask ktail_mask) {
            vmovups(fp32_inp | ktail_mask | T_z,
                    ptr[reg_inp + sizeof(float) * (idx)]);
            if (!mayiuse(avx512_core_bf16))
                bf16_emu_->r_vcvtneps2bf16(bf16_out, fp32_inp);
            else
                vcvtneps2bf16(bf16_out, fp32_inp);
            vmovdqu16(
                    yword[reg_out + sizeof(mkldnn_bfloat16_t) * (idx)] | ktail_mask,
                    bf16_out);
        };

        mov(reg_inp, ptr[abi_param1 + GET_OFF(inp)]);
        mov(reg_out, ptr[abi_param1 + GET_OFF(out)]);
        if (is_dynamic_size_)
            mov(reg_size, ptr[abi_param1 + GET_OFF(size)]);

        if (!mayiuse(avx512_core_bf16))
            bf16_emu_->init_vcvtneps2bf16();

        mov(reg32_tail, 0xffff);
        kmovw(ktail_mask, reg32_tail);

        if (is_dynamic_size_) { // determine size after JIT is called
            constexpr int n_unroll = 2; // unroll by powers of 2 from 2^n to 2^0
            Xbyak::Label l_simd_loop[n_unroll + 2], l_simd_notail;
            for (int i = n_unroll; i >= 0; i--) {
                const int unroll = 1 << i; // 4, 2, 1
                L(l_simd_loop[i + 1]); {
                    cmp(reg_size, simd_w_ * unroll);
                    jl(l_simd_loop[i], T_NEAR);
                    for (int j = 0; j < simd_w_ * unroll; j += simd_w_) {
                        cvt(j, ktail_mask);
                    }
                    add(reg_inp, simd_w_ * unroll * sizeof(float));
                    add(reg_out, simd_w_ * unroll * sizeof(mkldnn_bfloat16_t));
                    sub(reg_size, simd_w_ * unroll);
                    jmp(l_simd_loop[i + 1], T_NEAR);
                }
            }
            L(l_simd_loop[0]);
            test(reg_size, reg_size);
            jz(l_simd_notail);
            // JIT of `tail_mask_ = (1 << (size_ % simd_w_)) - 1;`
            mov(reg32_mask, 1);
            mov(reg64_tail, reg_size);
            shl(reg32_mask, reg8_mask_shift);
            sub(reg32_mask, 1);
            kmovd(ktail_mask, reg32_mask);
            cvt(0, ktail_mask);
            L(l_simd_notail);

        } else {

            size_t blocked_size = (size_ / simd_w_) * simd_w_;
            const size_t loop_length = 1024;
            const size_t number_of_loops = blocked_size / loop_length;
            const size_t tail_of_loops = blocked_size % loop_length;

            if (number_of_loops > 0) {
                Xbyak::Label l_number_of_loops;
                mov(reg_size, number_of_loops);
                L(l_number_of_loops);
                for (size_t i = 0; i < loop_length; i += simd_w_)
                    cvt(i, ktail_mask);
                add(reg_inp, sizeof(float) * loop_length);
                add(reg_out, sizeof(mkldnn_bfloat16_t) * loop_length);

                dec(reg_size);
                cmp(reg_size, 0);
                jg(l_number_of_loops, T_NEAR);
            }
            if (tail_of_loops > 0) {
                for (size_t i = 0; i < tail_of_loops; i += simd_w_)
                    cvt(i, ktail_mask);
                add(reg_inp, sizeof(float) * tail_of_loops);
                add(reg_out, sizeof(mkldnn_bfloat16_t) * tail_of_loops);
            }
            if (tail_mask_ != 0) {
                mov(reg32_tail, tail_mask_);
                kmovw(ktail_mask, reg32_tail);
                cvt(0, ktail_mask);
            }
        }
        postamble();
    }

    void (*jit_ker)(bf16_cvt_utils::jit_call_t *);

private:
    size_t size_;
    int tail_mask_;
    int simd_w_;

    bf16_emulation_t *bf16_emu_;
    bool is_dynamic_size_;

    Xbyak::Opmask ktail_mask = k2;
    Xbyak::Zmm fp32_inp = Xbyak::Zmm(0);
    Xbyak::Zmm fp32_tmp = Xbyak::Zmm(1);

    Xbyak::Zmm one = Xbyak::Zmm(2);
    Xbyak::Zmm even = Xbyak::Zmm(3);
    Xbyak::Zmm selector = Xbyak::Zmm(4);

    Xbyak::Ymm bf16_out = Xbyak::Ymm(5);

    Xbyak::Reg64 scratch = r15;
    Xbyak::Reg64 reg_inp = rax;
    Xbyak::Reg64 reg_out = rbx;
    Xbyak::Reg64 reg_size = rdx;

    Xbyak::Reg64 reg64_tail = rcx;
    Xbyak::Reg32 reg32_tail = ecx;
    Xbyak::Reg8 reg8_mask_shift = cl;
    Xbyak::Reg32 reg32_mask = r8d;

};

struct jit_avx512_core_cvt_bf16_to_ps_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_cvt_bf16_to_ps_t)

    jit_avx512_core_cvt_bf16_to_ps_t(void) : simd_w_(16), is_dynamic_size_(true) {
        generate();
        jit_ker = (void (*)(bf16_cvt_utils::jit_call_t *))getCode();
    }

    jit_avx512_core_cvt_bf16_to_ps_t(size_t size)
        : size_(size), simd_w_(16), is_dynamic_size_(false) {
        tail_mask_ = (1 << (size_ % simd_w_)) - 1;

        generate();
        jit_ker = (void (*)(bf16_cvt_utils::jit_call_t *))getCode();
    }

    void generate() {
        preamble();

        mov(reg_inp, ptr[abi_param1 + GET_OFF(inp)]);
        mov(reg_out, ptr[abi_param1 + GET_OFF(out)]);

        if (is_dynamic_size_) { // determine size after JIT is called
            mov(reg_size, ptr[abi_param1 + GET_OFF(size)]);
            constexpr int n_unroll = 2; // unroll by powers of 2 from 2^n to 2^0
            Xbyak::Label l_simd_loop[n_unroll + 2], l_simd_notail;
            for (int i = n_unroll; i >= 0; i--) {
                const int unroll = 1 << i; // 4, 2, 1
                L(l_simd_loop[i + 1]); {
                    cmp(reg_size, simd_w_ * unroll);
                    jl(l_simd_loop[i], T_NEAR);
                    for (int j = 0; j < simd_w_ * unroll; j += simd_w_) {
                        vpmovzxwd(zmm_cvt,
                                ptr[reg_inp + sizeof(mkldnn_bfloat16_t) * j]);
                        vpslld(zmm_cvt, zmm_cvt, 0x10);
                        vmovdqu32(zword[reg_out + sizeof(float) * j], zmm_cvt);
                    }
                    add(reg_inp, simd_w_ * unroll * sizeof(mkldnn_bfloat16_t));
                    add(reg_out, simd_w_ * unroll * sizeof(float));
                    sub(reg_size, simd_w_ * unroll);
                    jmp(l_simd_loop[i + 1], T_NEAR);
                }
            }
            L(l_simd_loop[0]);
            test(reg_size, reg_size);
            jz(l_simd_notail);
            // JIT of `tail_mask_ = (1 << (size_ % simd_w_)) - 1;`
            mov(reg32_mask, 1);
            mov(reg64_tail, reg_size);
            shl(reg32_mask, reg8_mask_shift);
            sub(reg32_mask, 1);
            kmovd(ktail_mask, reg32_mask);
            vpmovzxwd(zmm_cvt | ktail_mask | T_z, ptr[reg_inp]);
            vpslld(zmm_cvt, zmm_cvt, 0x10);
            vmovdqu32(zword[reg_out] | ktail_mask, zmm_cvt);
            L(l_simd_notail);

        } else {

            size_t blocked_size = (size_ / simd_w_) * simd_w_;
            const size_t loop_length = 1024;
            const size_t number_of_loops = blocked_size / loop_length;
            const size_t tail_of_loops = blocked_size % loop_length;

            if (number_of_loops > 0) {
                Xbyak::Label l_number_of_loops;
                mov(reg_size, number_of_loops);
                L(l_number_of_loops);
                for (size_t i = 0; i < loop_length; i += simd_w_) {
                    vpmovzxwd(zmm_cvt, ptr[reg_inp + sizeof(mkldnn_bfloat16_t) * i]);
                    vpslld(zmm_cvt, zmm_cvt, 0x10);
                    vmovups(zword[reg_out + sizeof(float) * i], zmm_cvt);
                }
                add(reg_inp, sizeof(mkldnn_bfloat16_t) * loop_length);
                add(reg_out, sizeof(float) * loop_length);

                dec(reg_size);
                cmp(reg_size, 0);
                jg(l_number_of_loops, T_NEAR);
            }

            if (tail_of_loops > 0) {
                for (size_t i = 0; i < tail_of_loops; i += simd_w_) {
                    vpmovzxwd(zmm_cvt, ptr[reg_inp + sizeof(mkldnn_bfloat16_t) * i]);
                    vpslld(zmm_cvt, zmm_cvt, 0x10);
                    vmovups(zword[reg_out + sizeof(float) * (i)], zmm_cvt);
                }
                add(reg_inp, sizeof(mkldnn_bfloat16_t) * tail_of_loops);
                add(reg_out, sizeof(float) * tail_of_loops);
            }
            if (tail_mask_ != 0) {
                mov(reg32_mask, tail_mask_);
                kmovw(ktail_mask, reg32_mask);

                vpmovzxwd(zmm_cvt | ktail_mask | T_z, ptr[reg_inp]);
                vpslld(zmm_cvt, zmm_cvt, 0x10);
                vmovups(zword[reg_out] | ktail_mask, zmm_cvt);
            }
        }

        postamble();
    }

    void (*jit_ker)(bf16_cvt_utils::jit_call_t *);

private:
    size_t size_;
    int tail_mask_;
    int simd_w_;
    bool is_dynamic_size_;

    Xbyak::Opmask ktail_mask = k1;
    Xbyak::Zmm zmm_cvt = Xbyak::Zmm(0);

    Xbyak::Reg64 reg_inp = rax;
    Xbyak::Reg64 reg_out = rbx;
    Xbyak::Reg64 reg_size = rdx;

    Xbyak::Reg64 reg64_tail = rcx;
    Xbyak::Reg32 reg32_tail = ecx;
    Xbyak::Reg8 reg8_mask_shift = cl;
    Xbyak::Reg32 reg32_mask = r8d;
};

// performs element-by-element sum of inp and add float arrays and stores
// result to bfloat16 out array with downconversion
struct jit_avx512_core_add_cvt_ps_to_bf16_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_add_cvt_ps_to_bf16)

    jit_avx512_core_add_cvt_ps_to_bf16_t() : simd_w_(16) {
        bf16_emu_ = new bf16_emulation_t(this, one, even,
                selector, scratch, fp32_tmp, fp32_tmp);

        generate();
        jit_ker = (void (*)(bf16_cvt_utils::jit_call_t *))getCode();
    }

    ~jit_avx512_core_add_cvt_ps_to_bf16_t() { delete bf16_emu_; }

    void generate() {
        preamble();

        auto add_cvt = [&](size_t idx, Xbyak::Opmask ktail_mask) {
            vmovups(fp32_inp | ktail_mask | T_z, ptr[reg_inp + sizeof(float) * (idx)]);
            vaddps(fp32_inp | ktail_mask | T_z, fp32_inp, ptr[reg_add + sizeof(float) * (idx)]);
            if (!mayiuse(avx512_core_bf16))
                bf16_emu_->r_vcvtneps2bf16(bf16_out, fp32_inp);
            else
                vcvtneps2bf16(bf16_out, fp32_inp);

            vmovdqu16(
                    yword[reg_out + sizeof(mkldnn_bfloat16_t) * (idx)] | ktail_mask,
                    bf16_out);
        };

        mov(reg_inp, ptr[abi_param1 + GET_OFF(inp)]);
        mov(reg_add, ptr[abi_param1 + GET_OFF(add)]);
        mov(reg_out, ptr[abi_param1 + GET_OFF(out)]);
        mov(reg_size, ptr[abi_param1 + GET_OFF(size)]);

        if (!mayiuse(avx512_core_bf16))
            bf16_emu_->init_vcvtneps2bf16();

        mov(reg32_tail, 0xffff);
        kmovw(ktail_mask, reg32_tail);

        constexpr int n_unroll = 2; // unroll by powers of 2 from 2^n to 2^0
        Xbyak::Label l_simd_loop[n_unroll + 2], l_simd_notail;
        for (int i = n_unroll; i >= 0; i--) {
            const int unroll = 1 << i; // 4, 2, 1
            L(l_simd_loop[i + 1]); {
                cmp(reg_size, simd_w_ * unroll);
                jl(l_simd_loop[i], T_NEAR);
                for (int j = 0; j < simd_w_ * unroll; j += simd_w_) {
                    add_cvt(j, ktail_mask);
                }
                add(reg_inp, simd_w_ * unroll * sizeof(float));
                add(reg_add, simd_w_ * unroll * sizeof(float));
                add(reg_out, simd_w_ * unroll * sizeof(mkldnn_bfloat16_t));

                sub(reg_size, simd_w_ * unroll);
                jmp(l_simd_loop[i + 1], T_NEAR);
            }
        }
        L(l_simd_loop[0]);
        test(reg_size, reg_size);
        jz(l_simd_notail);
        // JIT of `tail_mask_ = (1 << (size_ % simd_w_)) - 1;`
        mov(reg32_mask, 1);
        mov(reg64_tail, reg_size);
        shl(reg32_mask, reg8_mask_shift);
        sub(reg32_mask, 1);
        kmovd(ktail_mask, reg32_mask);
        add_cvt(0, ktail_mask);
        L(l_simd_notail);

        postamble();
    }

    void (*jit_ker)(bf16_cvt_utils::jit_call_t *);

private:
    int simd_w_;

    bf16_emulation_t *bf16_emu_;

    Xbyak::Opmask ktail_mask = k2;
    Xbyak::Zmm fp32_inp = Xbyak::Zmm(0);
    Xbyak::Zmm fp32_tmp = Xbyak::Zmm(1);

    Xbyak::Zmm one = Xbyak::Zmm(2);
    Xbyak::Zmm even = Xbyak::Zmm(3);
    Xbyak::Zmm selector = Xbyak::Zmm(4);
    Xbyak::Reg64 scratch = r15;

    Xbyak::Ymm bf16_out = Xbyak::Ymm(5);

    Xbyak::Reg64 reg_inp = rax;
    Xbyak::Reg64 reg_out = rbx;
    Xbyak::Reg64 reg_add = r11;
    Xbyak::Reg64 reg_size = rdx;

    Xbyak::Reg64 reg64_tail = rcx;
    Xbyak::Reg32 reg32_tail = ecx;
    Xbyak::Reg8 reg8_mask_shift = cl;
    Xbyak::Reg32 reg32_mask = r8d;
};

// implementation of reorder of part of tensor [s][16c] -> [S][16c][2s]
// it is required for quick implementation of 1x1 bf16 bwd_w jit kernel
// w/o using permw instruction inside
// TODO: consider modification/replacement for outer transformation jit kernel
struct jit_avx512_core_bf16_reorder_s16c_to_S16c2s_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_bf16_reorder_s16c_to_S16c2s)

    jit_avx512_core_bf16_reorder_s16c_to_S16c2s_t() : simd_w_(16) {
        generate();
        jit_ker = (void (*)(bf16_cvt_utils::jit_call_t *))getCode();
    }

    ~jit_avx512_core_bf16_reorder_s16c_to_S16c2s_t() {}

    void generate() {
        preamble();

        mov(reg_inp, ptr[abi_param1 + GET_OFF(inp)]);
        mov(reg_out, ptr[abi_param1 + GET_OFF(out)]);
        mov(reg_size, ptr[abi_param1 + GET_OFF(size)]);

        auto zmm_reg = [=](int idx) {
            assert(idx < 31);
            return Xbyak::Zmm(idx);
        };

        Xbyak::Label dst_prm_table;
        mov(reg_prm, dst_prm_table);
        vmovups(zmm_prm, ptr[reg_prm]);

        constexpr int n_unroll = 2; // unroll by powers of 2 from 2^n to 2^0
        int sizeofcacheline = 2 * simd_w_ * sizeof(mkldnn_bfloat16_t);
        Xbyak::Label l_simd_loop[n_unroll + 2], l_simd_notail;
        for (int i = n_unroll; i >= 0; i--) {
            const int unroll = 1 << i; // 4, 2, 1
            L(l_simd_loop[i + 1]); {
                cmp(reg_size, 2 * unroll);
                jl(l_simd_loop[i], T_NEAR);
                for (int j = 0; j < unroll; j++) {
                     auto zmm_inp = zmm_reg(j);
                     vmovups(zmm_inp, zword[reg_inp + j * sizeofcacheline]);
                     vpermw(zmm_inp, zmm_prm, zmm_inp);
                     vmovups(zword[reg_out + j * sizeofcacheline], zmm_inp);
                }
                add(reg_inp, unroll * sizeofcacheline);
                add(reg_out, unroll * sizeofcacheline);

                sub(reg_size, 2 * unroll);
                jmp(l_simd_loop[i + 1], T_NEAR);
            }
        }
        L(l_simd_loop[0]);

        test(reg_size, reg_size);
        jz(l_simd_notail);

        mov(reg32_tail, 0x00ff);
        kmovw(ktail_mask, reg32_tail);

        auto zmm_inp = zmm_reg(0);
        vpxord(zmm_inp, zmm_inp, zmm_inp);
        vmovups(zmm_inp | ktail_mask | T_z, ptr[reg_inp]);
        vpermw(zmm_inp, zmm_prm, zmm_inp);
        vmovups(zword[reg_out], zmm_inp);

        L(l_simd_notail);

        postamble();

        const uint16_t dst_prm_array[32] =
            {0,16,  1,17,  2,18,  3,19,  4,20,  5,21,  6,22,  7,23,  8,24,
            9,25,  10,26,  11,27,  12,28,  13,29,  14,30,  15,31 };

        align(64);
        L(dst_prm_table);
        for (size_t i = 0; i < 32; ++i)
            dw(dst_prm_array[i]);
    }

    void (*jit_ker)(bf16_cvt_utils::jit_call_t *);

private:
    int simd_w_;

    Xbyak::Opmask ktail_mask = k2;
    Xbyak::Zmm zmm_prm = Xbyak::Zmm(31);

    Xbyak::Reg64 reg_inp = rax;
    Xbyak::Reg64 reg_out = rbx;
    Xbyak::Reg64 reg_prm = r11;
    Xbyak::Reg64 reg_size = rdx;

    Xbyak::Reg32 reg32_tail = abi_not_param1.cvt32();
};

#undef GET_OFF
}
}
}

#endif
