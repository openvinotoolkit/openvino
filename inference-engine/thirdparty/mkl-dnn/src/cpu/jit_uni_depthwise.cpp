/*******************************************************************************
* Copyright 2018 Intel Corporation
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
        case alg_kind::depthwise_scale_shift: return 0;
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
    h->uni_vmulps(vmm_src, vmm_src, h->ptr[p_weights]);
    h->uni_vaddps(vmm_src, vmm_src, h->ptr[p_bias]);
}

template <cpu_isa_t isa>
void jit_uni_depthwise_injector_f32<isa>::prelu_compute_vector(const Vmm &vmm_src,
        const Xbyak::Reg64& p_weights, const Xbyak::Reg64& p_bias) {
    const unsigned char _cmp_gt_os = 6;
    const unsigned char _cmp_lt_os = 1;

    if (isa == sse42) {
        h->pxor(vmm_mask, vmm_mask);
        h->cmpps(vmm_mask, vmm_src, _cmp_gt_os);
        h->movups(vmm_aux0, vmm_src);
        h->mulps(vmm_aux0, h->ptr[p_weights]);
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

        bool isFlat = desc.src_desc.format == nchw && desc.dst_desc.format == nchw ;

        Reg64 param = abi_param1;

        const int block_size = isa == avx512_common ? 16 : 8;
        const int main_loop_step = isFlat ? block_size : 1;

        this->preamble();

        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_scale, ptr[param + GET_OFF(weights)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);
        if (with_bias_)
            mov(reg_shift, ptr[param + GET_OFF(bias)]);

        Label main_loop_label;
        Label tail_loop_label;
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
            jle(tail_loop_label, T_NEAR);

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
            uni_vmovups(xmm_dst, xmm_shift);
            uni_vfmadd231ps(xmm_dst, xmm_src, xmm_scale);
            movss(ptr[reg_to], xmm_dst);

            add(reg_from, 1*sizeof(float));
            add(reg_to, 1*sizeof(float));
            dec(reg_work_amount);

            jmp(tail_loop_label, T_NEAR);
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

        bool isFlat = desc.src_desc.format == nchw && desc.dst_desc.format == nchw;

        Reg64 param = abi_param1;

        const int block_size = isa == avx512_common ? 16 : 8;
        const int main_loop_step = isFlat ? block_size : 1;

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
        Label exit_label;

        L(main_loop_label); {
            cmp(reg_work_amount, main_loop_step-1);
            jle(tail_loop_label, T_NEAR);

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

            pxor(xmm_mask, xmm_mask);
            cmpps(xmm_mask, xmm_src, _cmp_gt_os);
            movups(xmm_dst, xmm_src);
            mulps(xmm_src, xmm_scale);
            blendvps(xmm_dst, xmm_src);

            movss(ptr[reg_to], xmm_dst);

            add(reg_from, 1*sizeof(float));
            add(reg_to, 1*sizeof(float));
            dec(reg_work_amount);

            jmp(tail_loop_label, T_NEAR);
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

    auto desired_blk_fmt = isa == avx512_common ? nChw16c : nChw8c;

    assert(engine()->kind() == engine_kind::cpu);
    bool ok = true && mayiuse(isa)
        && utils::one_of(desc()->prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference)
        && utils::everyone_is(data_type::f32, desc()->src_desc.data_type, desc()->dst_desc.data_type)
        && desc()->src_desc.format == desc()->dst_desc.format
        && utils::one_of(desc()->src_desc.format, desired_blk_fmt, nchw)
        && utils::one_of(desc()->dst_desc.format, desired_blk_fmt, nchw)
        && utils::one_of(desc()->weights_desc.format, x)
        && IMPLICATION(this->with_bias(), x == desc()->bias_desc.format)
        && attr()->has_default_values();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa>
jit_uni_depthwise_fwd_t<isa>::jit_uni_depthwise_fwd_t(const pd_t *pd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd), kernel_(nullptr),
      padded_weights_(nullptr), padded_bias_(nullptr) {
    const auto &desc = *conf_.desc();
    switch (desc.alg_kind) {
        case alg_kind::depthwise_scale_shift:
            kernel_ = new jit_uni_scale_shift_kernel_f32<isa>(desc, pd->with_bias()); break;
        case alg_kind::depthwise_prelu:
            kernel_ = new jit_uni_prelu_kernel_f32<isa>(desc, pd->with_bias()); break;
        default: assert(!"unknown depthwise alg_kind");
    }

    const int simd_w = isa == avx512_common ? 16 : 8;
    const memory_desc_wrapper data_d(conf_.src_pd());
    const int c_without_padding = data_d.dims()[1];
    const int c_padded = rnd_up(c_without_padding, simd_w);

    if (conf_.want_padded_weights()) {
        padded_weights_ = (data_t *)malloc(sizeof(data_t) * c_padded, 64);
        for (int oc = c_without_padding; oc < c_padded; ++oc)
            padded_weights_[oc] = 0;

        if (conf_.with_bias()) {
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
void jit_uni_depthwise_fwd_t<isa>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper data_d(conf_.src_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    const int N = data_d.dims()[0];
    const int C = data_d.dims()[1];
    const int H = data_d.dims()[2];
    const int W = data_d.dims()[3];

    const int simd_w = isa == avx512_common ? 16 : 8;
    const int ch_block_size = data_d.format() == nchw ? 1 : simd_w;
    const int CB = div_up(C, ch_block_size);

    if (conf_.want_padded_weights()) {
        for (int oc = 0; oc < C; ++oc)
            padded_weights_[oc] = weights[oc];
        weights = padded_weights_;

        if (conf_.with_bias()) {
            for (int oc = 0; oc < C; ++oc)
                padded_bias_[oc] = bias[oc];
            bias = padded_bias_;
        }
    }

    parallel_nd(N, CB, H,
        [&](int n, int cb, int h) {
        jit_args arg = {};

        arg.from    = &src[data_d.blk_off(n, cb, h)];
        arg.to      = &dst[data_d.blk_off(n, cb, h)];
        arg.weights = &weights[weights_d.blk_off(cb * ch_block_size)];
        if (bias)
            arg.bias = &bias[bias_d.blk_off(cb * ch_block_size)];
        arg.work_amount = (size_t)W;

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

            if (this->jcp.with_bias)
                uni_vmovups(vmm_acc, vmmword[reg_bias + i*4*sizeof(float)]);
            else
                uni_vpxor(vmm_acc, vmm_acc, vmm_acc);

            int o_off = ow*jcp.ch_block + i*4;
            if (this->jcp.with_sum)
                uni_vaddps(vmm_acc, vmm_acc,
                           vmmword[reg_output + o_off*sizeof(float)]);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::apply_filter(int ur_w, int kw_size) {
    int ch_blk = jcp.ch_block;
    int stride_w = jcp.stride_w;

    Label exit_label;

    int repeats = isa == sse42 ? 2 : 1;

    cmp(reg_kh, 1);
    jl(exit_label, T_NEAR);
    for (int i = 0; i < repeats; i++) {
        for (int kw = 0; kw < kw_size; kw++) {
            int ker_off = kw * ch_blk + i*4;

            Vmm vmm_ker = get_ker_reg(0);
            uni_vmovups(vmm_ker, ptr[aux_reg_kernel
                                     + ker_off * sizeof(float)]);

            for (int ow = 0; ow < ur_w; ow++) {
                int inp_off = ow * stride_w * ch_blk + kw * ch_blk + i*4;

                Vmm vmm_src = get_src_reg(0);
                uni_vmovups(vmm_src, ptr[aux_reg_input0
                                         + inp_off * sizeof(float)]);

                Vmm vmm_acc = get_acc_reg(i*ur_w + ow);
                uni_vfmadd231ps(vmm_acc, vmm_src, vmm_ker);
            }
        }
    }
    add(aux_reg_kernel, jcp.kw*ch_blk*sizeof(float));

    cmp(reg_kh, 2);
    jl(exit_label, T_NEAR);
    for (int i = 0; i < repeats; i++) {
        for (int kw = 0; kw < kw_size; kw++) {
            int ker_off = kw * ch_blk + i*4;

            Vmm vmm_ker = get_ker_reg(0);
            uni_vmovups(vmm_ker, ptr[aux_reg_kernel
                                     + ker_off * sizeof(float)]);

            for (int ow = 0; ow < ur_w; ow++) {
                int inp_off = ow * stride_w * ch_blk + kw * ch_blk + i*4;

                Vmm vmm_src = get_src_reg(0);
                uni_vmovups(vmm_src, ptr[aux_reg_input1
                                         + inp_off * sizeof(float)]);

                Vmm vmm_acc = get_acc_reg(i*ur_w + ow);
                uni_vfmadd231ps(vmm_acc, vmm_src, vmm_ker);
            }
        }
    }
    add(aux_reg_kernel, jcp.kw*ch_blk*sizeof(float));

    cmp(reg_kh, 3);
    jl(exit_label, T_NEAR);
    for (int i = 0; i < repeats; i++) {
        for (int kw = 0; kw < kw_size; kw++) {
            int ker_off = kw * ch_blk + i*4;

            Vmm vmm_ker = get_ker_reg(0);
            uni_vmovups(vmm_ker, ptr[aux_reg_kernel
                                     + ker_off * sizeof(float)]);

            for (int ow = 0; ow < ur_w; ow++) {
                int inp_off = ow * stride_w * ch_blk + kw * ch_blk + i*4;

                Vmm vmm_src = get_src_reg(0);
                uni_vmovups(vmm_src, ptr[aux_reg_input2
                                         + inp_off * sizeof(float)]);

                Vmm vmm_acc = get_acc_reg(i*ur_w + ow);
                uni_vfmadd231ps(vmm_acc, vmm_src, vmm_ker);
            }
        }
    }

    L(exit_label);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::apply_activation(int ur_w) {
    if (this->jcp.with_eltwise) {
        int repeats = isa == sse42 ? 2 : 1;
        eltwise_injector->compute_vector_range(4, repeats * ur_w + 4);
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::store_dst(int ur_w) {
    int repeats = isa == sse42 ? 2 : 1;
    for (int i = 0; i < repeats; i++) {
        for (int ow = 0; ow < ur_w; ow++) {
            int o_off = ow*jcp.ch_block + i*4;
            Vmm vmm_dst = get_acc_reg(i*ur_w + ow);

            uni_vmovups(vmmword[reg_output + o_off*sizeof(float)], vmm_dst);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::loop_body() {
    Label left_pad_label;
    Label right_pad_label;
    Label unrolled_w_label;
    Label tail_w_label;
    Label exit_label;

    L(left_pad_label); {
        int ur_w = 1;
        int kw = jcp.iw == 1 ? jcp.kw - 2 : jcp.kw - 1;

        mov(aux_reg_input0, reg_input0);
        mov(aux_reg_input1, reg_input1);
        mov(aux_reg_input2, reg_input2);
        mov(aux_reg_kernel, reg_kernel);
        add(aux_reg_kernel, jcp.ch_block*sizeof(float));

        load_src(ur_w);
        apply_filter(ur_w, kw);
        apply_activation(ur_w);
        store_dst(ur_w);

        add(reg_input0, sizeof(float) * ur_w * jcp.ch_block * (jcp.stride_w-1));
        add(reg_input1, sizeof(float) * ur_w * jcp.ch_block * (jcp.stride_w-1));
        add(reg_input2, sizeof(float) * ur_w * jcp.ch_block * (jcp.stride_w-1));

        add(reg_output, sizeof(float) * ur_w * jcp.ch_block);

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
        apply_activation(ur_w);
        store_dst(ur_w);

        add(reg_input0, sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_input1, sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_input2, sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_output, sizeof(float) * ur_w * jcp.ch_block);

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
        apply_activation(ur_w);
        store_dst(ur_w);

        add(reg_input0, sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_input1, sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_input2, sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_output, sizeof(float) * ur_w * jcp.ch_block);

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
            apply_activation(ur_w);
            store_dst(ur_w);

            sub(reg_ur_w, ur_w);
        }
    }

    L(exit_label);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_row_f32<isa>::generate()
{
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

    loop_body();

    this->postamble();

    if (jcp.with_eltwise)
        eltwise_injector->prepare_table();
}

template <cpu_isa_t isa>
status_t jit_uni_dw_conv_row_f32<isa>::init_conf(jit_conv_conf_t &jcp,
        int ic, int ih, int iw, int oh, int ow, int ker_h, int ker_w, int str_h, int str_w, alg_kind_t eltwise_alg,
        float eltwise_alpha, float eltwise_beta, bool with_sum) {
    if (!mayiuse(isa)) return status::unimplemented;
    const int simd_w = isa == avx512_common ? 16 : 8;

    jcp.kh = ker_h;
    jcp.kw = ker_w;
    jcp.ch_block = simd_w;
    jcp.with_bias = true;
    jcp.ic = ic;
    jcp.oc = ic;
    jcp.ih = ih;
    jcp.iw = iw;
    jcp.oh = oh;
    jcp.ow = ow;
    jcp.stride_h = str_h;
    jcp.stride_w = str_w;

    if (jcp.kh != 3 || jcp.kw != 3)
        return  status::unimplemented;

    jcp.ur_w = 4;

    jcp.with_eltwise  = eltwise_alg != mkldnn_alg_kind_undef;
    jcp.eltwise_alg   = eltwise_alg;
    jcp.eltwise_alpha = eltwise_alpha;
    jcp.eltwise_beta  = eltwise_beta;
    jcp.with_sum = with_sum;

    return status::success;
}

template struct jit_uni_dw_conv_row_f32<avx512_common>;
template struct jit_uni_dw_conv_row_f32<avx2>;
template struct jit_uni_dw_conv_row_f32<sse42>;

}
}
}
