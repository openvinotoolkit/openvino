/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
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

#include <mkldnn_types.h>
#include "mkldnn_types.h"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "utils.hpp"
#include "jit_generator.hpp"

#include "jit_uni_depthwise.hpp"

#define GET_OFF(field) offsetof(jit_args, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

struct jit_args {
    const float *from;
    const float *to;
    const float *weights;
    const float *bias;
    size_t work_amount;
};

struct jit_uni_depthwise_kernel_f32 : public c_compatible {
    const depthwise_desc_t &desc_;
    void (*ker_)(const jit_args *);
    bool with_bias_;

    void operator()(const jit_args *args) { assert(ker_); ker_(args); }

    jit_uni_depthwise_kernel_f32(const depthwise_desc_t &desc, bool with_bias)
        : desc_(desc), ker_(nullptr), with_bias_(with_bias) {}
    virtual ~jit_uni_depthwise_kernel_f32() {}
};

template <cpu_isa_t isa>
int jit_uni_depthwise_injector_f32<isa>::aux_vecs_count(alg_kind_t depthwise_alg) {
    switch (depthwise_alg) {
        case alg_kind::depthwise_scale_shift: return isa == sse42 ? 1 : 0;
        case alg_kind::depthwise_prelu: return 2;
        default: assert(!"unsupported depthwise algorithm");
    }

    return 0;
}

template <cpu_isa_t isa>
void jit_uni_depthwise_injector_f32<isa>::injector_preamble(size_t start_idx, size_t end_idx) {
    preserved_vecs_count = 0;
    vecs_to_preserve = (size_t)jit_uni_depthwise_injector_f32<isa>::aux_vecs_count(depthwise_alg);

    for (size_t i = 0; i < vecs_count; i++) {
        if (preserved_vecs_count >= vecs_to_preserve)
            break;

        if (i < start_idx || i >= end_idx) {
            preserved_vec_idxs[preserved_vecs_count] = i;
            preserved_vecs_count++;
        }
    }

    start_idx_tail = start_idx;
    size_t preserved_vecs_count_tail = vecs_to_preserve - preserved_vecs_count;
    for (size_t i = 0; i < preserved_vecs_count_tail; i++) {
        preserved_vec_idxs[preserved_vecs_count] = start_idx + i;
        preserved_vecs_count++;
        start_idx_tail = start_idx + i + 1;
    }

    h->sub(h->rsp, preserved_vecs_count * vlen);
    for (size_t i = 0; i < preserved_vecs_count; ++i)
        h->uni_vmovups(h->ptr[h->rsp + i * vlen], Vmm(preserved_vec_idxs[i]));

    assign_regs();
}

template <cpu_isa_t isa>
void jit_uni_depthwise_injector_f32<isa>::injector_preamble_tail(size_t start_idx, size_t end_idx) {
    size_t tail_vecs_to_preserve = start_idx_tail - start_idx;
    int idx_off = (vecs_to_preserve - tail_vecs_to_preserve);

    if (tail_vecs_to_preserve > 0) {
        h->add(h->rsp, idx_off * vlen);
        for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
            h->uni_vmovups(Vmm(preserved_vec_idxs[idx_off + i]), h->ptr[h->rsp + i * vlen]);

        for (size_t i = 0; i < tail_vecs_to_preserve; ++i) {
            preserved_vec_idxs[idx_off + i] += tail_vecs_to_preserve;
        }

        for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
            h->uni_vmovups(h->ptr[h->rsp + i * vlen], Vmm(preserved_vec_idxs[idx_off + i]));
        h->sub(h->rsp, idx_off * vlen);

        assign_regs();
    }
}

template <cpu_isa_t isa>
void jit_uni_depthwise_injector_f32<isa>::injector_postamble() {
    for (size_t i = 0; i < preserved_vecs_count; ++i)
        h->uni_vmovups(Vmm(preserved_vec_idxs[i]), h->ptr[h->rsp + i * vlen]);
    h->add(h->rsp, preserved_vecs_count * vlen);
}

template <cpu_isa_t isa>
void jit_uni_depthwise_injector_f32<isa>::assign_regs() {
    vmm_mask = Vmm(preserved_vec_idxs[0]);
    vmm_aux0 = Vmm(preserved_vec_idxs[1]);
}

template <cpu_isa_t isa>
void jit_uni_depthwise_injector_f32<isa>::scale_shift_compute_vector(const Vmm &vmm_src,
        const Xbyak::Reg64& p_weights, const Xbyak::Reg64& p_bias) {
    if (isa == sse42) {
        h->movups(vmm_mask, h->ptr[p_weights]);
        h->mulps(vmm_src, vmm_mask);
        h->movups(vmm_mask, h->ptr[p_bias]);
        h->addps(vmm_src, vmm_mask);
    } else {
        h->uni_vmulps(vmm_src, vmm_src, h->ptr[p_weights]);
        h->uni_vaddps(vmm_src, vmm_src, h->ptr[p_bias]);
    };
}

template <cpu_isa_t isa>
void jit_uni_depthwise_injector_f32<isa>::prelu_compute_vector(const Vmm &vmm_src,
        const Xbyak::Reg64& p_weights, const Xbyak::Reg64& p_bias) {
    const unsigned char _cmp_gt_os = 6;
    const unsigned char _cmp_lt_os = 1;

    if (isa == sse42) {
        h->pxor(vmm_mask, vmm_mask);
        h->cmpps(vmm_mask, vmm_src, _cmp_gt_os);
        h->movups(vmm_aux0, h->ptr[p_weights]);
        h->mulps(vmm_aux0, vmm_src);
        h->blendvps(vmm_src, vmm_aux0);
    } else if (isa == avx2) {
        h->vxorps(vmm_mask, vmm_mask, vmm_mask);
        h->vcmpgtps(vmm_mask, vmm_src, vmm_mask);
        h->vmulps(vmm_aux0, vmm_src, h->ptr[p_weights]);
        h->vblendvps(vmm_src, vmm_aux0, vmm_src, vmm_mask);
    } else if (isa == avx512_common) {
        h->vxorpd(vmm_mask, vmm_mask, vmm_mask);
        h->vmovups(vmm_aux0, vmm_src);
        h->vcmpps(k_mask, vmm_src, vmm_mask, _cmp_lt_os);
        h->vmulps(vmm_src | k_mask, vmm_aux0, h->ptr[p_weights]);
    }
}

template <cpu_isa_t isa>
void jit_uni_depthwise_injector_f32<isa>::compute_body(size_t start_idx, size_t end_idx,
        const Xbyak::Reg64& p_weights, const Xbyak::Reg64& p_bias) {
    for (size_t idx = start_idx; idx < end_idx; idx++) {
        switch (depthwise_alg) {
            case alg_kind::depthwise_scale_shift:
                scale_shift_compute_vector(Vmm(idx), p_weights, p_bias); break;
            case alg_kind::depthwise_prelu:
                prelu_compute_vector(Vmm(idx), p_weights, p_bias); break;
            default: assert(!"unsupported depthwise algorithm");
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_depthwise_injector_f32<isa>::compute_vector_range(int start_idx, int end_idx,
        const Xbyak::Reg64& p_weights, const Xbyak::Reg64& p_bias) {
    injector_preamble(start_idx, end_idx);
    compute_body(start_idx_tail, end_idx, p_weights, p_bias);
    injector_preamble_tail(start_idx, end_idx);
    compute_body(start_idx, start_idx_tail, p_weights, p_bias);
    injector_postamble();
}

template struct jit_uni_depthwise_injector_f32<avx512_common>;
template struct jit_uni_depthwise_injector_f32<avx2>;
template struct jit_uni_depthwise_injector_f32<sse42>;

/* jit kernels */
namespace {

template <cpu_isa_t isa>
struct jit_uni_scale_shift_kernel_f32 : public jit_uni_depthwise_kernel_f32,
    public jit_generator
{
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_scale_shift_kernel_f32)
    jit_uni_scale_shift_kernel_f32(const depthwise_desc_t &desc, bool with_bias)
        : jit_uni_depthwise_kernel_f32(desc, with_bias), jit_generator() {
        assert(desc.alg_kind == alg_kind::depthwise_scale_shift);
        assert(isa == sse42 || isa == avx2 || isa == avx512_common);

        bool isFlat = (desc.src_desc.format == nchw && desc.dst_desc.format == nchw) ||
                      (desc.src_desc.format == ncdhw && desc.dst_desc.format == ncdhw);

        Reg64 param = abi_param1;

        const int block_size = isa == avx512_common ? 16 : 8;
        const int main_loop_step = (isFlat || desc.src_desc.format == nc) ? block_size : 1;

        this->preamble();

        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_scale, ptr[param + GET_OFF(weights)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);
        if (with_bias_)
            mov(reg_shift, ptr[param + GET_OFF(bias)]);

        Label main_loop_label;
        Label tail_loop_label;
        Label tail_loop_flat_label;
        Label exit_label;

        int repeats = isa == sse42 ? 2 : 1;
        for (int i = 0; i < repeats; i++) {
            if (isFlat) {
                uni_vbroadcastss(get_scale_reg(i), ptr[reg_scale]);
                if (with_bias_)
                    uni_vbroadcastss(get_shift_reg(i), ptr[reg_shift]);
                else
                    uni_vpxor(get_shift_reg(i), get_shift_reg(i), get_shift_reg(i));
            } else {
                uni_vmovups(get_scale_reg(i), ptr[reg_scale + i*4*sizeof(float)]);
                if (with_bias_)
                    uni_vmovups(get_shift_reg(i), ptr[reg_shift + i*4*sizeof(float)]);
                else
                    uni_vpxor(get_shift_reg(i), get_shift_reg(i), get_shift_reg(i));
            }
        }

        if (isFlat) {
            uni_vbroadcastss(xmm_scale, ptr[reg_scale]);
            if (with_bias_)
                uni_vbroadcastss(xmm_shift, ptr[reg_shift]);
            else
                uni_vpxor(xmm_shift, xmm_shift, xmm_shift);
        }

        L(main_loop_label); {
            cmp(reg_work_amount, main_loop_step-1);
            jle(isFlat ? tail_loop_flat_label : tail_loop_label, T_NEAR);

            int repeats = isa == sse42 ? 2 : 1;
            for (int i = 0; i < repeats; i++) {
                uni_vmovups(vmm_src, ptr[reg_from + i*4*sizeof(float)]);
                uni_vmovups(vmm_dst, get_shift_reg(i));
                uni_vfmadd231ps(vmm_dst, vmm_src, get_scale_reg(i));
                uni_vmovups(ptr[reg_to + i*4*sizeof(float)], vmm_dst);
            }

            add(reg_from, block_size*sizeof(float));
            add(reg_to, block_size*sizeof(float));
            sub(reg_work_amount, main_loop_step);

            jmp(main_loop_label, T_NEAR);
        }

        L(tail_loop_label); {
            cmp(reg_work_amount, 0);
            jle(exit_label, T_NEAR);

            movss(xmm_src, ptr[reg_from]);
            movss(xmm_shift, ptr[reg_shift]);
            movss(xmm_scale, ptr[reg_scale]);
            uni_vmovups(xmm_dst, xmm_shift);
            uni_vfmadd231ps(xmm_dst, xmm_src, xmm_scale);
            movss(ptr[reg_to], xmm_dst);

            add(reg_from, 1*sizeof(float));
            add(reg_to, 1*sizeof(float));
            add(reg_shift, 1*sizeof(float));
            add(reg_scale, 1*sizeof(float));
            dec(reg_work_amount);

            jmp(tail_loop_label, T_NEAR);
        }

        L(tail_loop_flat_label); {
            cmp(reg_work_amount, 0);
            jle(exit_label, T_NEAR);

            movss(xmm_src, ptr[reg_from]);
            uni_vmovups(xmm_dst, xmm_shift);
            uni_vfmadd231ps(xmm_dst, xmm_src, xmm_scale);
            movss(ptr[reg_to], xmm_dst);

            add(reg_from, 1*sizeof(float));
            add(reg_to, 1*sizeof(float));
            dec(reg_work_amount);

            jmp(tail_loop_flat_label, T_NEAR);
        }

        L(exit_label);

        this->postamble();

        ker_ = (decltype(ker_))this->getCode();
    }

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
                                             isa == avx2, Ymm, Zmm>::type;

    inline Vmm get_scale_reg(int idx) { return Vmm(idx + 2); }
    inline Vmm get_shift_reg(int idx) { return Vmm(idx + 4); }

    Reg64 reg_from = r8;
    Reg64 reg_to = r9;
    Reg64 reg_work_amount = r10;
    Reg64 reg_scale = r11;
    Reg64 reg_shift = r12;

    Vmm vmm_src = Vmm(0);
    Vmm vmm_dst = Vmm(1);

    Xmm xmm_src = Xmm(0);
    Xmm xmm_dst = Xmm(1);
    Xmm xmm_scale = Xmm(6);
    Xmm xmm_shift = Xmm(7);
};

template <cpu_isa_t isa>
struct jit_uni_prelu_kernel_f32 : public jit_uni_depthwise_kernel_f32,
    public jit_generator
{
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_prelu_kernel_f32)
    jit_uni_prelu_kernel_f32(const depthwise_desc_t &desc, bool with_bias)
        : jit_uni_depthwise_kernel_f32(desc, with_bias), jit_generator() {
        assert(desc.alg_kind == alg_kind::depthwise_prelu);
        assert(isa == sse42 || isa == avx2 || isa == avx512_common);

        bool isFlat = (desc.src_desc.format == nchw && desc.dst_desc.format == nchw) ||
                      (desc.src_desc.format == ncdhw && desc.dst_desc.format == ncdhw);

        Reg64 param = abi_param1;

        const int block_size = isa == avx512_common ? 16 : 8;
        const int main_loop_step = (isFlat || desc.src_desc.format == nc) ? block_size : 1;

        this->preamble();

        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_scale, ptr[param + GET_OFF(weights)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        int repeats = isa == sse42 ? 2 : 1;
        for (int i = 0; i < repeats; i++) {
            if (isFlat) {
                uni_vbroadcastss(get_scale_reg(i), ptr[reg_scale]);
            } else {
                uni_vmovups(get_scale_reg(i), ptr[reg_scale + i*4*sizeof(float)]);
            }
        }

        if (isFlat) {
            uni_vbroadcastss(xmm_scale, ptr[reg_scale]);
        }

        Label main_loop_label;
        Label tail_loop_label;
        Label tail_loop_flat_label;
        Label exit_label;

        L(main_loop_label); {
            cmp(reg_work_amount, main_loop_step-1);
            jle(isFlat ? tail_loop_flat_label :tail_loop_label, T_NEAR);

            for (int i = 0; i < repeats; i++) {
                uni_vmovups(vmm_src, ptr[reg_from + i*4*sizeof(float)]);

                if (isa == sse42) {
                    pxor(vmm_mask, vmm_mask);
                    cmpps(vmm_mask, vmm_src, _cmp_gt_os);
                    movups(vmm_dst, vmm_src);
                    mulps(vmm_src, get_scale_reg(i));
                    blendvps(vmm_dst, vmm_src);
                } else if (isa == avx2) {
                    vcmpgtps(vmm_mask, vmm_src, vmm_zero);
                    vmulps(vmm_dst, vmm_src, get_scale_reg(i));
                    vblendvps(vmm_dst, vmm_dst, vmm_src, vmm_mask);
                } else if (isa == avx512_common) {
                    Opmask kmask = Opmask(7);
                    vmovups(vmm_dst, vmm_src);
                    vcmpps(kmask, vmm_src, vmm_zero, _cmp_lt_os);
                    vmulps(vmm_dst | kmask, vmm_src, get_scale_reg(i));
                }

                uni_vmovups(ptr[reg_to + i*4*sizeof(float)], vmm_dst);
            }

            add(reg_from, block_size*sizeof(float));
            add(reg_to, block_size*sizeof(float));
            sub(reg_work_amount, main_loop_step);

            jmp(main_loop_label, T_NEAR);
        }

        L(tail_loop_label); {
            cmp(reg_work_amount, 0);
            jle(exit_label, T_NEAR);

            movss(xmm_src, ptr[reg_from]);
            movss(xmm_scale, ptr[reg_scale]);

            pxor(xmm_mask, xmm_mask);
            cmpps(xmm_mask, xmm_src, _cmp_gt_os);
            movups(xmm_dst, xmm_src);
            mulps(xmm_src, xmm_scale);
            blendvps(xmm_dst, xmm_src);

            movss(ptr[reg_to], xmm_dst);

            add(reg_from, 1*sizeof(float));
            add(reg_to, 1*sizeof(float));
            add(reg_scale, 1*sizeof(float));
            dec(reg_work_amount);

            jmp(tail_loop_label, T_NEAR);
        }

        L(tail_loop_flat_label); {
            cmp(reg_work_amount, 0);
            jle(exit_label, T_NEAR);

            movss(xmm_src, ptr[reg_from]);

            pxor(xmm_mask, xmm_mask);
            cmpps(xmm_mask, xmm_src, _cmp_gt_os);
            movups(xmm_dst, xmm_src);
            mulps(xmm_src, xmm_scale);
            blendvps(xmm_dst, xmm_src);

            movss(ptr[reg_to], xmm_dst);

            add(reg_from, 1*sizeof(float));
            add(reg_to, 1*sizeof(float));
            dec(reg_work_amount);

            jmp(tail_loop_flat_label, T_NEAR);
        }

        L(exit_label);

        this->postamble();

        ker_ = (decltype(ker_))this->getCode();
    }

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
                                             isa == avx2, Ymm, Zmm>::type;

    inline Vmm get_scale_reg(int idx) { return Vmm(idx + 4); }

    Reg64 reg_from = r8;
    Reg64 reg_to = r9;
    Reg64 reg_work_amount = r10;
    Reg64 reg_scale = r11;

    Vmm vmm_mask = Vmm(0);
    Vmm vmm_src = Vmm(1);
    Vmm vmm_zero = Vmm(2);
    Vmm vmm_dst = Vmm(3);

    Xmm xmm_mask = Xmm(0);
    Xmm xmm_src = Xmm(1);
    Xmm xmm_dst = Xmm(3);
    Xmm xmm_scale = Xmm(4);

    const unsigned char _cmp_gt_os = 6;
    const unsigned char _cmp_lt_os = 1;
};

} /* namespace */

template <cpu_isa_t isa>
status_t jit_uni_depthwise_fwd_t<isa>::pd_t::init() {
    using namespace alg_kind;

    memory_format_t desired_blk_fmt, desired_pln_fmt;
    if (desc()->src_desc.ndims == 5) {
        desired_blk_fmt = isa == avx512_common ? nCdhw16c : nCdhw8c;
        desired_pln_fmt = ncdhw;
    } else if (desc()->src_desc.ndims == 4) {
        desired_blk_fmt = isa == avx512_common ? nChw16c : nChw8c;
        desired_pln_fmt = nchw;
    } else {
        desired_blk_fmt = nc;
        desired_pln_fmt = nc;
    }

    assert(engine()->kind() == engine_kind::cpu);
    bool ok = true && mayiuse(isa)
        && utils::one_of(desc()->prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference)
        && utils::everyone_is(data_type::f32, desc()->src_desc.data_type, desc()->dst_desc.data_type)
        && desc()->src_desc.format == desc()->dst_desc.format
        && utils::one_of(desc()->src_desc.format, desired_blk_fmt, desired_pln_fmt)
        && utils::one_of(desc()->dst_desc.format, desired_blk_fmt, desired_pln_fmt)
        && utils::one_of(desc()->weights_desc.format, x)
        && IMPLICATION(this->with_bias(), x == desc()->bias_desc.format)
        && attr()->has_default_values();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa>
jit_uni_depthwise_fwd_t<isa>::jit_uni_depthwise_fwd_t(const pd_t *apd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(apd, inputs, outputs), kernel_(nullptr),
      padded_weights_(nullptr), padded_bias_(nullptr) {
    const auto &desc = *pd()->desc();
    switch (desc.alg_kind) {
        case alg_kind::depthwise_scale_shift:
            kernel_ = new jit_uni_scale_shift_kernel_f32<isa>(desc, pd()->with_bias()); break;
        case alg_kind::depthwise_prelu:
            kernel_ = new jit_uni_prelu_kernel_f32<isa>(desc, pd()->with_bias()); break;
        default: assert(!"unknown depthwise alg_kind");
    }

    const int simd_w = isa == avx512_common ? 16 : 8;
    const memory_desc_wrapper data_d(pd()->src_pd());
    const int c_without_padding = data_d.dims()[1];
    const int c_padded = rnd_up(c_without_padding, simd_w);

    if (pd()->want_padded_weights()) {
        padded_weights_ = (data_t *)malloc(sizeof(data_t) * c_padded, 64);
        for (int oc = c_without_padding; oc < c_padded; ++oc)
            padded_weights_[oc] = 0;

        if (pd()->with_bias()) {
            padded_bias_ = (data_t *)malloc(sizeof(data_t) * c_padded, 64);
            for (int oc = c_without_padding; oc < c_padded; ++oc)
                padded_bias_[oc] = 0;
        }
    }
}

template <cpu_isa_t isa>
jit_uni_depthwise_fwd_t<isa>::~jit_uni_depthwise_fwd_t() {
    delete kernel_;
    free(padded_weights_);
    free(padded_bias_);
}

template <cpu_isa_t isa>
void jit_uni_depthwise_fwd_t<isa>::execute_forward() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper data_d(pd()->src_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));
    const memory_desc_wrapper bias_d(pd()->weights_pd(1));

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int D = pd()->D();
    const int H = pd()->H();
    const int W = pd()->W();

    const int simd_w = isa == avx512_common ? 16 : 8;
    const int ch_block_size = (data_d.format() == nchw) || (data_d.format() == ncdhw) ? 1 : simd_w;
    const int CB = div_up(C, ch_block_size);

    if (pd()->want_padded_weights()) {
        for (int oc = 0; oc < C; ++oc)
            padded_weights_[oc] = weights[oc];
        weights = padded_weights_;

        if (pd()->with_bias()) {
            for (int oc = 0; oc < C; ++oc)
                padded_bias_[oc] = bias[oc];
            bias = padded_bias_;
        }
    }

    parallel_nd(MB, CB, D, H,
        [&](int mb, int cb, int d, int h) {
        auto arg = jit_args();

        size_t data_off = data_d.ndims() == 4
                          ? data_d.blk_off(mb, cb, h)
                          : data_d.ndims() == 5
                            ? data_d.blk_off(mb, cb, d, h)
                            : data_d.blk_off(mb, cb * ch_block_size);

        arg.from    = &src[data_off];
        arg.to      = &dst[data_off];
        arg.weights = &weights[weights_d.blk_off(cb * ch_block_size)];
        if (bias)
            arg.bias = &bias[bias_d.blk_off(cb * ch_block_size)];
        arg.work_amount = data_d.format() == nc ? nstl::min(ch_block_size, C - cb * ch_block_size) : (size_t)W;

        (*kernel_)(&arg);
    });
}

template struct jit_uni_depthwise_fwd_t<sse42>;
template struct jit_uni_depthwise_fwd_t<avx2>;
template struct jit_uni_depthwise_fwd_t<avx512_common>;


#define GET_OFF_DW(field) offsetof(jit_conv_call_s, field)

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::load_src(int ur_w) {
    int repeats = isa == sse42 ? 2 : 1;
    for (int i = 0; i < repeats; i++) {
        for (int ow = 0; ow < ur_w; ow++) {
            Vmm vmm_acc = get_acc_reg(i*ur_w + ow);

            uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::apply_filter(int ur_w, int kw_size) {
    auto load_src = [=](Vmm vmm_src, const Xbyak::Address &op) {
        if (jcp.src_dt == data_type::u8) {
            uni_vpmovzxbd(vmm_src, op);
        } else {
            uni_vmovups(vmm_src, op);
        }
    };

    auto load_ker = [=](Vmm vmm_ker, const Xbyak::Address &op) {
        if (jcp.src_dt == data_type::u8) {
            uni_vpmovsxbd(vmm_ker, op);
        } else {
            uni_vmovups(vmm_ker, op);
        }
    };

    auto compute = [=](Vmm vmm_acc, Vmm vmm_src, Vmm vmm_ker) {
        if (jcp.src_dt == data_type::u8) {
            uni_vpmulld(vmm_src, vmm_src, vmm_ker);
            uni_vpaddd(vmm_acc, vmm_acc, vmm_src);
        } else {
            uni_vfmadd231ps(vmm_acc, vmm_src, vmm_ker);
        }
    };

    int ch_blk = jcp.ch_block;
    int stride_w = jcp.stride_w;

    Label exit_label;

    int repeats = isa == sse42 ? 2 : 1;

    cmp(reg_kh, 1);
    jl(exit_label, T_NEAR);
    for (int i = 0; i < repeats; i++) {
        for (int kw = 0; kw < kw_size; kw++) {
            int ker_off = kw * ch_blk + i*(jcp.ch_block / 2);

            Vmm vmm_ker = get_ker_reg(0);
            load_ker(vmm_ker, ptr[aux_reg_kernel + ker_off * jcp.typesize_in]);

            for (int ow = 0; ow < ur_w; ow++) {
                int inp_off = ow * stride_w * ch_blk + kw * ch_blk + i*(jcp.ch_block / 2);

                Vmm vmm_src = get_src_reg(0);
                load_src(vmm_src, ptr[aux_reg_input0 + inp_off * jcp.typesize_in]);

                Vmm vmm_acc = get_acc_reg(i*ur_w + ow);
                compute(vmm_acc, vmm_src, vmm_ker);
            }
        }
    }
    add(aux_reg_kernel, jcp.kw*ch_blk*jcp.typesize_in);

    cmp(reg_kh, 2);
    jl(exit_label, T_NEAR);
    for (int i = 0; i < repeats; i++) {
        for (int kw = 0; kw < kw_size; kw++) {
            int ker_off = kw * ch_blk + i*(jcp.ch_block / 2);

            Vmm vmm_ker = get_ker_reg(0);
            load_ker(vmm_ker, ptr[aux_reg_kernel + ker_off * jcp.typesize_in]);

            for (int ow = 0; ow < ur_w; ow++) {
                int inp_off = ow * stride_w * ch_blk + kw * ch_blk + i*(jcp.ch_block / 2);

                Vmm vmm_src = get_src_reg(0);
                load_src(vmm_src, ptr[aux_reg_input1 + inp_off * jcp.typesize_in]);

                Vmm vmm_acc = get_acc_reg(i*ur_w + ow);
                compute(vmm_acc, vmm_src, vmm_ker);
            }
        }
    }
    add(aux_reg_kernel, jcp.kw*ch_blk*jcp.typesize_in);

    cmp(reg_kh, 3);
    jl(exit_label, T_NEAR);
    for (int i = 0; i < repeats; i++) {
        for (int kw = 0; kw < kw_size; kw++) {
            int ker_off = kw * ch_blk + i*(jcp.ch_block / 2);

            Vmm vmm_ker = get_ker_reg(0);
            load_ker(vmm_ker, ptr[aux_reg_kernel + ker_off * jcp.typesize_in]);

            for (int ow = 0; ow < ur_w; ow++) {
                int inp_off = ow * stride_w * ch_blk + kw * ch_blk + i*(jcp.ch_block / 2);

                Vmm vmm_src = get_src_reg(0);
                load_src(vmm_src, ptr[aux_reg_input2 + inp_off * jcp.typesize_in]);

                Vmm vmm_acc = get_acc_reg(i*ur_w + ow);
                compute(vmm_acc, vmm_src, vmm_ker);
            }
        }
    }

    L(exit_label);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::cvt2ps(data_type_t type_in, Vmm vmm_in, const Operand &op, bool scalar_load) {
    Xmm xmm_in = Xmm(vmm_in.getIdx());

    switch (type_in) {
        case data_type::f32:
        case data_type::s32:
            if (scalar_load) {
                mov(reg_tmp_32, op);
                movq(xmm_in, reg_tmp_64);
            } else {
                uni_vmovups(vmm_in, op);
            }
            break;
        case data_type::s8:
            if (scalar_load) {
                movsx(reg_tmp_32, op);
                movq(xmm_in, reg_tmp_64);
            } else {
                uni_vpmovsxbd(vmm_in, op);
            }
            break;
        case data_type::u8:
            if (scalar_load) {
                movzx(reg_tmp_32, op);
                movq(xmm_in, reg_tmp_64);
            } else {
                uni_vpmovzxbd(vmm_in, op);
            }
            break;
        default: assert(!"unsupported data type");
    }

    if (type_in != data_type::f32)
        uni_vcvtdq2ps(vmm_in, vmm_in);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::apply_postprocessing(int ur_w, int oc_step) {
    int repeats = isa == sse42 ? 2 : 1;

    for (int r = 0; r < repeats; r++) {
        for (int ow = 0; ow < ur_w; ow++) {
            if (jcp.src_dt == data_type::u8) {
                uni_vcvtdq2ps(get_acc_reg(r * ur_w + ow), get_acc_reg(r * ur_w + ow));
            }

            if (jcp.with_bias) {
                int b_off = r * (jcp.ch_block / 2);
                cvt2ps(jcp.bia_dt, vmm_bias, ptr[reg_bias + b_off * jcp.typesize_bia], false);
                uni_vaddps(get_acc_reg(r * ur_w + ow), get_acc_reg(r * ur_w + ow), vmm_bias);
            }
        }
    }

    if (jcp.with_sum) {
        for (int r = 0; r < repeats; r++) {
            int tail_size = isa == sse42 ? nstl::min(jcp.ch_block / 2, oc_step - r * jcp.ch_block / 2) : oc_step;
            bool is_scalar_store = isa == sse42 ? tail_size < jcp.ch_block / 2 : tail_size < jcp.ch_block;

            for (int ow = 0; ow < ur_w; ow++) {
                if (is_scalar_store) {
                    if (isa == avx512_common) {
                        int o_off = ow * ow_stride_;

                        Vmm vmm_in = vmm_sum | ktail_mask | T_z;

                        cvt2ps(jcp.dst_dt, vmm_in, ptr[reg_output + o_off * jcp.typesize_out], false);
                        uni_vaddps(get_acc_reg(r * ur_w + ow), get_acc_reg(r * ur_w + ow), vmm_sum);
                    } else {
                        for (int oc = 0; oc < tail_size; oc++) {
                            int o_off = ow * ow_stride_ + r * (jcp.ch_block / 2) + oc;

                            uni_vpxor(vmm_sum, vmm_sum, vmm_sum);
                            cvt2ps(jcp.dst_dt, vmm_sum, ptr[reg_output + o_off * jcp.typesize_out], true);

                            if (oc >= jcp.ch_block / 2) {
                                vperm2i128(Ymm(vmm_sum.getIdx()), Ymm(vmm_sum.getIdx()), Ymm(vmm_sum.getIdx()), 0x01);
                            }
                            uni_vpslldq(vmm_sum, vmm_sum, jcp.typesize_out * (oc % (jcp.ch_block / 2)));

                            uni_vaddps(get_acc_reg(r * ur_w + ow), get_acc_reg(r * ur_w + ow), vmm_sum);
                        }
                    }
                } else {
                    int o_off = ow * ow_stride_ + r * (jcp.ch_block / 2);

                    uni_vpxor(vmm_sum, vmm_sum, vmm_sum);
                    cvt2ps(jcp.dst_dt, vmm_sum, ptr[reg_output + o_off * jcp.typesize_out], false);

                    uni_vaddps(get_acc_reg(r * ur_w + ow), get_acc_reg(r * ur_w + ow), vmm_sum);
                }
            }
        }
    }

    const auto &p = attr_.post_ops_;
    int eltwise_inj_idx = 0;
    int depthwise_inj_idx = 0;
    int start_idx = p.find(primitive_kind::convolution) + 1;
    for (int i = start_idx; i < p.len_; i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors[eltwise_inj_idx]->compute_vector_range(4, 4 + repeats * ur_w);
            eltwise_inj_idx++;
        } else if (post_op.is_depthwise()) {
            mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data));
            mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data));

            add(reg_d_weights, reg_oc_off);
            add(reg_d_bias, reg_oc_off);

            depthwise_injectors[depthwise_inj_idx]->compute_vector_range(4, 4 + ur_w, reg_d_weights, reg_d_bias);

            if (repeats == 2) {
                add(reg_d_weights, (jcp.ch_block / 2) * sizeof(float));
                add(reg_d_bias, (jcp.ch_block / 2) * sizeof(float));

                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(4 + ur_w, 4 + 2 * ur_w, reg_d_weights, reg_d_bias);
            }

            depthwise_inj_idx++;
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::store_dst_typed(const Xbyak::Address &op, Vmm vmm_dst, bool scalar_store) {
    Ymm ymm_dst = Ymm(vmm_dst.getIdx());
    Xmm xmm_dst = Xmm(vmm_dst.getIdx());

    switch (jcp.dst_dt) {
        case data_type::f32:
        case data_type::s32:
            if (scalar_store) {
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_32);
            } else {
                uni_vmovups(op, vmm_dst);
            }
            break;
        case data_type::s8:
            uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);

            if (isa != sse42 && !scalar_store)
                vpermq(ymm_dst, ymm_dst, 0x08);

            uni_vpacksswb(vmm_dst, vmm_dst, vmm_dst);

            if (scalar_store) {
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
            } else {
                if (isa != sse42)
                    vmovq(op, xmm_dst);
                else
                    movd(op, xmm_dst);
            }
            break;
        case data_type::u8:
        case data_type::bin:
            uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);

            if (isa != sse42 && !scalar_store)
                vpermq(ymm_dst, ymm_dst, 0x08);

            uni_vpackuswb(vmm_dst, vmm_dst, vmm_dst);

            if (scalar_store) {
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
            } else {
                if (isa != sse42)
                    vmovq(op, xmm_dst);
                else
                    movd(op, xmm_dst);
            }
            break;
        default:
            assert(!"unknown dst_dt");
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::store_dst(int ur_w, int oc_step) {
    int nbits = 8;
    int repeats = isa == sse42 && oc_step > (jcp.ch_block / 2) ? 2 : 1;

    if (isa == avx512_common && oc_step != jcp.ch_block) {
        int mask = (1 << oc_step) - 1;
        mov(reg_tmp_32, mask);
        kmovw(ktail_mask, reg_tmp_32);
    }

    for (int i = 0; i < repeats; i++) {
        for (int ow = 0; ow < ur_w; ow++) {
            Vmm vmm_dst = get_acc_reg(i * ur_w + ow);
            if (jcp.dst_dt != data_type::f32 && jcp.dst_dt != data_type::bin) {
                if (attr_.round_mode_ == round_mode::nearest)
                    uni_vcvtps2dq(vmm_dst, vmm_dst);
                else if (attr_.round_mode_ == round_mode::down) {
                    uni_vroundps(vmm_dst, vmm_dst, 1);
                    uni_vcvtps2dq(vmm_dst, vmm_dst);
                } else
                    assert(!"unimplemented");
            }
        }
    }

    if (jcp.with_binarization) {
        int output_step = div_up(ow_stride_, nbits);

        const auto &p = attr_.post_ops_;
        int binarization_idx = p.find(primitive_kind::binarization);

        push(reg_bias);

        mov(reg_b_weights, reinterpret_cast<size_t>(p.entry_[binarization_idx].binarization.weights_data));
        mov(reg_b_out_mask, reinterpret_cast<size_t>(p.entry_[binarization_idx].binarization.output_mask_data));
        add(reg_b_weights, reg_oc_off);
        add(reg_b_out_mask, reg_oc_off);

        for (int ow = 0; ow < ur_w; ow++) {
            for (int i = 0; i < repeats; i++) {
                int tail_size = isa == sse42 ? nstl::min(jcp.ch_block / 2, oc_step - i * jcp.ch_block / 2) : oc_step;
                mov(reg_b_mask, (1 << tail_size) - 1);
                uni_vmovups(vmm_thr, ptr[reg_b_weights + i * (jcp.ch_block / 2) * sizeof(float)]);
                uni_vmovups(vmm_out_mask, ptr[reg_b_out_mask + i * (jcp.ch_block / 2) * sizeof(float)]);

                Vmm vmm_dst = get_acc_reg(i * ur_w + ow);

                if (isa == avx512_common) {
                    vcmpps(bin_mask0, vmm_dst, vmm_thr, _cmp_gt_os);
                    vptestmd(bin_mask1, vmm_out_mask, vmm_out_mask);
                    kxnorw(bin_mask0, bin_mask0, bin_mask1);
                } else {
                    uni_vcmpgtps(vmm_dst, vmm_dst, vmm_thr);
                    uni_vpcmpeqd(vmm_dst, vmm_dst, vmm_out_mask);
                }

                if (i == 0) {
                    if (isa == avx512_common) {
                        kmovw(reg_tmp_32, bin_mask0);
                    } else {
                        uni_vmovmskps(reg_tmp_32, vmm_dst);
                    }
                    and_(reg_tmp_64, reg_b_mask);
                } else {
                    uni_vmovmskps(reg_tmp2_32, vmm_dst);
                    and_(reg_tmp2_64, reg_b_mask);
                    shl(reg_tmp2_32, 4);
                    or_(reg_tmp_32, reg_tmp2_32);
                }

                if (i == repeats - 1) {
                    const size_t o_off = ow * output_step;
                    if (isa == avx512_common && oc_step > nbits) {
                        mov(ptr[reg_output + o_off * jcp.typesize_out], reg_tmp_16);
                    } else {
                        mov(ptr[reg_output + o_off * jcp.typesize_out], reg_tmp_8);
                    }
                }
            }
        }

        pop(reg_bias);
    } else {
        for (int i = 0; i < repeats; i++) {
            int tail_size = isa == sse42 ? nstl::min(jcp.ch_block / 2, oc_step - i * jcp.ch_block / 2) : oc_step;
            bool is_scalar_store = isa == sse42 ? tail_size < jcp.ch_block / 2 : tail_size < jcp.ch_block;
            if (is_scalar_store) {
                for (int ow = 0; ow < ur_w; ow++) {
                    Vmm vmm_dst = get_acc_reg(i * ur_w + ow);

                    if (isa == avx512_common) {
                        int o_off = ow * ow_stride_;

                        store_dst_typed(ptr[reg_output + o_off * jcp.typesize_out], vmm_dst | ktail_mask, false);
                    } else {
                        for (int oc = 0; oc < tail_size; oc++) {
                            int o_off = ow * ow_stride_ + i * (jcp.ch_block / 2) + oc;
                            store_dst_typed(ptr[reg_output + o_off * jcp.typesize_out], vmm_dst, true);

                            if (isa == sse42) {
                                psrldq(vmm_dst, jcp.typesize_out);
                            } else {
                                Ymm ymm_dst = Ymm(vmm_dst.getIdx());

                                vperm2i128(ymm_tmp, ymm_dst, ymm_dst, 0x01);
                                vpalignr(ymm_dst, vmm_tmp, ymm_dst, jcp.typesize_out);
                            }
                        }
                    }
                }
            } else {
                for (int ow = 0; ow < ur_w; ow++) {
                    int o_off = ow * ow_stride_ + i * (jcp.ch_block / 2);
                    Vmm vmm_dst = get_acc_reg(i * ur_w + ow);

                    store_dst_typed(ptr[reg_output + o_off * jcp.typesize_out], vmm_dst, false);
                }
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::loop_body(int oc_step) {
    Label left_pad_label;
    Label right_pad_label;
    Label unrolled_w_label;
    Label tail_w_label;
    Label exit_label;

    int output_step = jcp.with_binarization ? div_up(ow_stride_, 8) : ow_stride_;

    L(left_pad_label); {
        int ur_w = 1;
        int kw = jcp.iw == 1 ? jcp.kw - 2 : jcp.kw - 1;

        mov(aux_reg_input0, reg_input0);
        mov(aux_reg_input1, reg_input1);
        mov(aux_reg_input2, reg_input2);
        mov(aux_reg_kernel, reg_kernel);
        add(aux_reg_kernel, jcp.ch_block*jcp.typesize_in);

        load_src(ur_w);
        apply_filter(ur_w, kw);
        apply_postprocessing(ur_w, oc_step);
        store_dst(ur_w, oc_step);

        add(reg_input0, jcp.typesize_in * ur_w * jcp.ch_block * (jcp.stride_w-1));
        add(reg_input1, jcp.typesize_in * ur_w * jcp.ch_block * (jcp.stride_w-1));
        add(reg_input2, jcp.typesize_in * ur_w * jcp.ch_block * (jcp.stride_w-1));
        add(reg_output, jcp.typesize_out * ur_w * output_step);

        sub(reg_ur_w, ur_w);
    }

    L(unrolled_w_label); {
        int ur_w = jcp.ur_w;
        int kw = jcp.kw;

        cmp(reg_ur_w, ur_w);
        jle(tail_w_label, T_NEAR);

        mov(aux_reg_input0, reg_input0);
        mov(aux_reg_input1, reg_input1);
        mov(aux_reg_input2, reg_input2);
        mov(aux_reg_kernel, reg_kernel);

        load_src(ur_w);
        apply_filter(ur_w, kw);
        apply_postprocessing(ur_w, oc_step);
        store_dst(ur_w, oc_step);

        add(reg_input0, jcp.typesize_in * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_input1, jcp.typesize_in * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_input2, jcp.typesize_in * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_output, jcp.typesize_out * ur_w * output_step);

        sub(reg_ur_w, ur_w);
        jmp(unrolled_w_label, T_NEAR);
    }

    L(tail_w_label); {
        int ur_w = 1;
        int kw = jcp.kw;

        cmp(reg_ur_w, ur_w);
        if (jcp.ow > 1)
            jle(right_pad_label, T_NEAR);
        else
            jle(exit_label, T_NEAR);

        mov(aux_reg_input0, reg_input0);
        mov(aux_reg_input1, reg_input1);
        mov(aux_reg_input2, reg_input2);
        mov(aux_reg_kernel, reg_kernel);

        load_src(ur_w);
        apply_filter(ur_w, kw);
        apply_postprocessing(ur_w, oc_step);
        store_dst(ur_w, oc_step);

        add(reg_input0, jcp.typesize_in * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_input1, jcp.typesize_in * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_input2, jcp.typesize_in * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_output, jcp.typesize_out * ur_w * output_step);

        sub(reg_ur_w, ur_w);
        jmp(tail_w_label, T_NEAR);
    }

    if (jcp.ow > 1) {
        L(right_pad_label); {
            int ur_w = 1;
            int kw = jcp.kw - ((jcp.stride_w == 1) ? 1 : jcp.iw % jcp.stride_w);

            mov(aux_reg_input0, reg_input0);
            mov(aux_reg_input1, reg_input1);
            mov(aux_reg_input2, reg_input2);
            mov(aux_reg_kernel, reg_kernel);

            load_src(ur_w);
            apply_filter(ur_w, kw);
            apply_postprocessing(ur_w, oc_step);
            store_dst(ur_w, oc_step);

            sub(reg_ur_w, ur_w);
        }
    }

    L(exit_label);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::generate() {
    const auto &p = attr_.post_ops_;
    int start_idx = p.find(primitive_kind::convolution) + 1;
    for (int i = start_idx; i < p.len_; i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors.push_back(new jit_uni_eltwise_injector_f32<isa>(
                    this,
                    post_op.eltwise.alg,
                    post_op.eltwise.alpha,
                    post_op.eltwise.beta
            ));
        } else if (post_op.is_depthwise()) {
            depthwise_injectors.push_back(new jit_uni_depthwise_injector_f32<isa>(
                    this,
                    post_op.depthwise.alg
            ));
        }
    }

    this->preamble();

    mov(reg_input0, ptr[this->param1 + GET_OFF_DW(src_row0)]);
    mov(reg_input1, ptr[this->param1 + GET_OFF_DW(src_row1)]);
    mov(reg_input2, ptr[this->param1 + GET_OFF_DW(src_row2)]);
    mov(reg_output, ptr[this->param1 + GET_OFF_DW(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF_DW(filt)]);
    if (jcp.with_bias)
        mov(reg_bias, ptr[this->param1 + GET_OFF_DW(bias)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF_DW(kh_padding)]);
    mov(reg_ur_w, ptr[this->param1 + GET_OFF_DW(ur_w)]);
    mov(reg_oc_work, ptr[this->param1 + GET_OFF_DW(oc_work)]);
    mov(reg_oc_off, ptr[this->param1 + GET_OFF_DW(oc_off)]);

    Label(tail_label);
    Label(exit_label);

    cmp(reg_oc_work, jcp.ch_block);
    jl(tail_label, T_NEAR);

    loop_body(jcp.ch_block);
    jmp(exit_label, T_NEAR);

    L(tail_label);

    if (jcp.oc % jcp.ch_block != 0)
        loop_body(jcp.oc % jcp.ch_block);

    L(exit_label);

    this->postamble();

    for (auto& inj : eltwise_injectors)
        inj->prepare_table();
}

template <cpu_isa_t isa>
bool jit_uni_dw_conv_row_f32<isa>::post_ops_ok(jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    int start_idx = p.find(primitive_kind::convolution) + 1;

    auto all_post_ops_supported = [&]() {
        bool ok = true;

        for (int i = start_idx; i < p.len_; i++) {
            ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::sum, primitive_kind::eltwise, primitive_kind::depthwise,
                                                       primitive_kind::binarization);
        }
        return ok;
    };
    auto contain = [&](mkldnn::impl::primitive_kind_t kind) { return p.find(kind, start_idx, -1) != -1; };
    auto position = [&](mkldnn::impl::primitive_kind_t kind) { return p.find(kind, start_idx, -1); };
    auto count = [&](mkldnn::impl::primitive_kind_t kind) { return p.count(kind, start_idx, -1); };

    return all_post_ops_supported() &&
           count(primitive_kind::sum) <= 1 &&
           count(primitive_kind::binarization) <= 1 &&
           IMPLICATION(contain(primitive_kind::sum), position(primitive_kind::sum) == start_idx) &&
           IMPLICATION(contain(primitive_kind::binarization), position(primitive_kind::binarization) == p.len_-1) &&
           IMPLICATION(contain(primitive_kind::binarization), !contain(primitive_kind::sum));
}

template <cpu_isa_t isa>
status_t jit_uni_dw_conv_row_f32<isa>::init_conf(jit_1x1_conv_conf_t &jcp, jit_conv_conf_t &jcp_dw,
        const primitive_attr_t &attr) {
    if (!mayiuse(isa)) return status::unimplemented;
    const int simd_w = isa == avx512_common ? 16 : 8;

    const auto &p = attr.post_ops_;

    int dw_conv_ind = p.find(primitive_kind::convolution);
    jcp_dw.with_sum = p.find(primitive_kind::sum, dw_conv_ind) != -1;

    jcp_dw.ch_block = simd_w;
    jcp_dw.with_bias = true;

    jcp_dw.kh = p.entry_[dw_conv_ind].dw_conv.ker_h;
    jcp_dw.kw = p.entry_[dw_conv_ind].dw_conv.ker_w;
    jcp_dw.ic = jcp.oc;
    jcp_dw.oc = jcp.oc;
    jcp_dw.ih = p.entry_[dw_conv_ind].dw_conv.in_h;
    jcp_dw.iw = p.entry_[dw_conv_ind].dw_conv.in_w;
    jcp_dw.oh = jcp.dw_conv_oh;
    jcp_dw.ow = jcp.dw_conv_ow;
    jcp_dw.stride_h = p.entry_[dw_conv_ind].dw_conv.str_h;
    jcp_dw.stride_w = p.entry_[dw_conv_ind].dw_conv.str_w;
    jcp_dw.conv_weights = p.entry_[dw_conv_ind].dw_conv.weights_data;
    jcp_dw.conv_biases = p.entry_[dw_conv_ind].dw_conv.biases_data;

    if (jcp_dw.kh != 3 || jcp_dw.kw != 3)
        return status::unimplemented;

    if (!post_ops_ok(jcp_dw, attr))
        return status::unimplemented;

    jcp_dw.ur_w = 4;

    jcp_dw.src_dt = jcp.dst_dt;
    jcp_dw.dst_dt = jcp.dw_conv_dst_dt;
    jcp_dw.bia_dt = jcp.bia_dt;
    jcp_dw.typesize_in = (int)types::data_type_size(jcp_dw.src_dt);
    jcp_dw.typesize_bia = (int)types::data_type_size(jcp_dw.bia_dt);
    jcp_dw.typesize_out = (int)types::data_type_size(jcp_dw.dst_dt);

    if (jcp_dw.src_dt != mkldnn_f32 && jcp_dw.src_dt != mkldnn_u8)
        return status::unimplemented;

    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_dw_conv_row_f32<isa>::init_conf(jit_conv_conf_t &jcp, jit_conv_conf_t &jcp_dw,
        const primitive_attr_t &attr) {
    if (!mayiuse(isa)) return status::unimplemented;
    const int simd_w = isa == avx512_common ? 16 : 8;

    const auto &p = attr.post_ops_;

    int dw_conv_ind = p.find(primitive_kind::convolution);
    jcp_dw.with_sum = p.find(primitive_kind::sum, dw_conv_ind) != -1;

    jcp_dw.ch_block = simd_w;
    jcp_dw.with_bias = true;

    jcp_dw.kh = p.entry_[dw_conv_ind].dw_conv.ker_h;
    jcp_dw.kw = p.entry_[dw_conv_ind].dw_conv.ker_w;
    jcp_dw.ic = jcp.oc;
    jcp_dw.oc = jcp.oc;
    jcp_dw.ih = p.entry_[dw_conv_ind].dw_conv.in_h;
    jcp_dw.iw = p.entry_[dw_conv_ind].dw_conv.in_w;
    jcp_dw.oh = jcp.dw_conv_oh;
    jcp_dw.ow = jcp.dw_conv_ow;
    jcp_dw.stride_h = p.entry_[dw_conv_ind].dw_conv.str_h;
    jcp_dw.stride_w = p.entry_[dw_conv_ind].dw_conv.str_w;
    jcp_dw.conv_weights = p.entry_[dw_conv_ind].dw_conv.weights_data;
    jcp_dw.conv_biases = p.entry_[dw_conv_ind].dw_conv.biases_data;

    if (jcp_dw.kh != 3 || jcp_dw.kw != 3)
        return status::unimplemented;

    if (!post_ops_ok(jcp_dw, attr))
        return status::unimplemented;

    jcp_dw.ur_w = 4;

    jcp_dw.src_dt = jcp.dst_dt;
    jcp_dw.dst_dt = jcp.dw_conv_dst_dt;
    jcp_dw.bia_dt = jcp.bia_dt;
    jcp_dw.typesize_in = (int)types::data_type_size(jcp_dw.src_dt);
    jcp_dw.typesize_bia = (int)types::data_type_size(jcp_dw.bia_dt);
    jcp_dw.typesize_out = (int)types::data_type_size(jcp_dw.dst_dt);

    if (jcp_dw.src_dt != mkldnn_f32 && jcp_dw.src_dt != mkldnn_u8)
        return status::unimplemented;

    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_dw_conv_row_f32<isa>::init_conf(jit_bin_conv_conf_t &jcp, jit_conv_conf_t &jcp_dw,
        const primitive_attr_t &attr) {
    if (!mayiuse(isa)) return status::unimplemented;
    const int simd_w = isa == avx512_common ? 16 : 8;

    const auto &p = attr.post_ops_;

    int dw_conv_ind = p.find(primitive_kind::convolution);
    jcp_dw.with_sum = p.find(primitive_kind::sum, dw_conv_ind) != -1;
    jcp_dw.with_binarization = p.find(primitive_kind::binarization, dw_conv_ind) != -1;

    jcp_dw.ch_block = simd_w;
    jcp_dw.with_bias = true;

    jcp_dw.kh = p.entry_[dw_conv_ind].dw_conv.ker_h;
    jcp_dw.kw = p.entry_[dw_conv_ind].dw_conv.ker_w;
    jcp_dw.ic = jcp.oc;
    jcp_dw.oc = jcp.oc;
    jcp_dw.ih = p.entry_[dw_conv_ind].dw_conv.in_h;
    jcp_dw.iw = p.entry_[dw_conv_ind].dw_conv.in_w;
    jcp_dw.oh = jcp.dw_conv_oh;
    jcp_dw.ow = jcp.dw_conv_ow;
    jcp_dw.stride_h = p.entry_[dw_conv_ind].dw_conv.str_h;
    jcp_dw.stride_w = p.entry_[dw_conv_ind].dw_conv.str_w;
    jcp_dw.conv_weights = p.entry_[dw_conv_ind].dw_conv.weights_data;
    jcp_dw.conv_biases = p.entry_[dw_conv_ind].dw_conv.biases_data;

    if (jcp_dw.kh != 3 || jcp_dw.kw != 3)
        return status::unimplemented;

    if (!post_ops_ok(jcp_dw, attr))
        return status::unimplemented;

    jcp_dw.ur_w = 4;

    jcp_dw.src_dt = mkldnn_f32;
    jcp_dw.dst_dt = jcp_dw.with_binarization ? mkldnn_bin : mkldnn_f32;
    jcp_dw.bia_dt = mkldnn_f32;
    jcp_dw.typesize_in = (int)types::data_type_size(jcp_dw.src_dt);
    jcp_dw.typesize_bia = (int)types::data_type_size(jcp_dw.bia_dt);
    jcp_dw.typesize_out = (int)types::data_type_size(jcp_dw.dst_dt);

    if (jcp_dw.src_dt != mkldnn_f32 && jcp_dw.src_dt != mkldnn_u8)
        return status::unimplemented;

    return status::success;
}

template struct jit_uni_dw_conv_row_f32<avx512_common>;
template struct jit_uni_dw_conv_row_f32<avx2>;
template struct jit_uni_dw_conv_row_f32<sse42>;

}
}
}
