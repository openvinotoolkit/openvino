/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#ifndef JIT_UNI_1x1_CONV_UTILS_HPP
#define JIT_UNI_1x1_CONV_UTILS_HPP

#include "mkldnn_thread.hpp"
#include "utils.hpp"
#include "nstl.hpp"

#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;

/* 1x1-kernel does not support non-unit strides so far, so the idea is:
 *  - for fwd or bwd_weights: to copy src to a scratch memory (with strides
 *    equal to 1) and then call the kernel
 *  - for bwd_data: reduce the problem to the one with unit stride by
 *    performing computations in a scratch memory (with strides equal to 1)
 *    and then copy the result to diff_src */
template <typename conv_pd_t>
inline void rtus_prepare(conv_pd_t *self, const convolution_desc_t *&conv_d,
        const memory_desc_t *&src_d, const memory_desc_t *dst_d) {
    const bool is_bwd_data = self->cdesc()->prop_kind
        == prop_kind::backward_data;

    const int ndims = src_d->ndims;
    bool rtus_applicable = true
        && utils::pick(ndims - 3,
            (conv_d->strides[0] != 1 && !one_of(conv_d->src_desc.data_type,
                data_type::s16, data_type::s32)),
            (conv_d->strides[0] != 1 || conv_d->strides[1] != 1))
        && utils::one_of(src_d->format, memory_format::nCw8c,
            memory_format::nCw16c, memory_format::nChw8c,
            memory_format::nChw16c);
    for (int d = 2; d < ndims; ++d) {
        /* TODO: relax these conditions (by improving reducer) */
        rtus_applicable = rtus_applicable
            && conv_d->padding[0][d - 2] == 0
            && dst_d->dims[d] * conv_d->strides[d - 2] == src_d->dims[d];
    }

    if (rtus_applicable) {
        self->rtus_.reduce_src_ = true;
        conv_d = &(self->rtus_.conv_d_ = *conv_d);
        self->rtus_.conv_d_.strides[0] = 1;
        if (ndims == 4)
            self->rtus_.conv_d_.strides[1] = 1;
        utils::array_set(self->rtus_.conv_d_.padding[0], 0, 2);
        if (ndims == 4)
            utils::array_set(self->rtus_.conv_d_.padding[1], 0, 2);
        const int ic = src_d->dims[1];
        if (is_bwd_data) {
            src_d = &(self->rtus_.conv_d_.diff_src_desc = *dst_d);
            self->rtus_.conv_d_.diff_src_desc.dims[1] = ic;
            memory_desc_wrapper::compute_blocking(
                    self->rtus_.conv_d_.diff_src_desc);
        } else {
            data_type_t data_type = self->rtus_.conv_d_.src_desc.data_type;
            src_d = &(self->rtus_.conv_d_.src_desc = *dst_d);
            self->rtus_.conv_d_.src_desc.dims[1] = ic;
            self->rtus_.conv_d_.src_desc.data_type = data_type;
            memory_desc_wrapper::compute_blocking(
                    self->rtus_.conv_d_.src_desc);
        }
    }
}

template <cpu_isa_t isa>
struct rtus_driver_t: public jit_generator {

    struct call_params_t {
        const void *ws; /* reduced image (w/ strides = 1) */
        const void *src; /* source image (w/ non-unit strides) */
        size_t icb;
        size_t os;
        size_t iw_start;
    };

    void (*ker_)(const call_params_t *p);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(rtus_driver_t)

    /* cpu specific part */
    using Vmm = typename utils::conditional<isa == avx2, Xbyak::Ymm,
          Xbyak::Zmm>::type;

    Xbyak::Reg64 reg_ws = abi_param1;
    Xbyak::Reg64 reg_src = abi_not_param1;
    Xbyak::Reg64 reg_icb = rdx;
    Xbyak::Reg64 reg_os = r11;
    Xbyak::Reg64 reg_iw_start = r8;

    Xbyak::Reg64 reg_cur_os = rax;
    Xbyak::Reg64 reg_cur_iw = r9;
    Xbyak::Reg64 reg_cur_src = r10;

    int iw_, stride_w_;
    int src_step_h_, src_step_icb_, ws_step_icb_, vlen_, vlen_shift_;
    bool src_to_ws_;
    size_t typesize_;
    Vmm reg_zero;
    Vmm reg_v;

    rtus_driver_t(int iw, int stride_w, int src_step_h,
            int src_step_icb, int ws_step_icb, bool src_to_ws, size_t typesize)
        : iw_(iw), stride_w_(stride_w), src_step_h_(src_step_h)
        , src_step_icb_(src_step_icb), ws_step_icb_(ws_step_icb)
        , src_to_ws_(src_to_ws), typesize_(typesize)
    {
        using namespace Xbyak;
        vlen_ = cpu_isa_traits<isa>::vlen;
        vlen_shift_ = cpu_isa_traits<isa>::vlen_shift;
        if (typesize_ == 2) {
            vlen_ /= 2;
            vlen_shift_--;
        }

        reg_zero = Vmm(0);
        reg_v = Vmm(1);

        generate();
    }

    void loop_is() {
        using namespace Xbyak;

        mov(reg_cur_src, reg_src);
        mov(reg_cur_iw, reg_iw_start);
        mov(reg_cur_os, reg_os);

        Label is_loop, skip_h_step;
        L(is_loop);

        if (src_to_ws_) {
            vmovups(reg_v, ptr[reg_cur_src]);
            vmovups(ptr[reg_ws], reg_v);
        } else {
            vmovups(reg_v, ptr[reg_ws]);
            vmovups(ptr[reg_cur_src], reg_v);
            for (int w = 1; w < stride_w_; ++w)
                vmovups(ptr[reg_cur_src + w * vlen_], reg_zero);
        }

        add(reg_ws, vlen_);

        add(reg_cur_iw, stride_w_);
        add(reg_cur_src, stride_w_ * vlen_);

        cmp(reg_cur_iw, iw_);
        jl(skip_h_step);
        /* for 1d convolution the loop over h should be skipped */
        if (src_step_icb_ == iw_) jmp(skip_h_step);

        if (src_to_ws_) {
            add(reg_cur_src, (src_step_h_ - iw_) * vlen_);
        } else {
            Xbyak::Reg64 reg_cur_src_fin = reg_cur_iw; /* just reuse */
            mov(reg_cur_src_fin, reg_cur_src);
            add(reg_cur_src_fin, (src_step_h_ - iw_) * vlen_);
            Label ih_loop;
            L(ih_loop);

            for (int w = 0; w < stride_w_; ++w)
                vmovups(ptr[reg_cur_src + w * vlen_], reg_zero);

            add(reg_cur_src, stride_w_ * vlen_);
            cmp(reg_cur_src, reg_cur_src_fin);
            jl(ih_loop);
        }
        xor_(reg_cur_iw, reg_cur_iw);

        L(skip_h_step);

        sub(reg_cur_os, vlen_);
        jnz(is_loop);

        /* restore dst */
        sub(reg_ws, reg_os);
    }

    void generate() {
        using namespace Xbyak;
        assert(isa == avx2 || isa == avx512_common
                || isa == avx512_core || isa == avx512_mic);

#if defined(_WIN32)
        assert(reg_src == abi_not_param1 && abi_not_param1 == rdi);
        push(rdi);
#endif

#define READ_PARAM(what) \
        mov(reg_ ## what, ptr[abi_param1 + offsetof(call_params_t, what)])
        READ_PARAM(src);
        READ_PARAM(icb);
        READ_PARAM(os);
        READ_PARAM(iw_start);

        assert(reg_ws == abi_param1);
        READ_PARAM(ws); /* reg_ws should always be read the last */
#undef  READ_PARAM

        shl(reg_os, vlen_shift_);

        if (!src_to_ws_)
            uni_vpxor(reg_zero, reg_zero, reg_zero);

        Label icb_loop;
        L(icb_loop);

        loop_is();

        add(reg_ws, ws_step_icb_ * vlen_);
        add(reg_src, src_step_icb_ * vlen_);

        dec(reg_icb);
        jnz(icb_loop, T_NEAR);

#if defined(_WIN32)
        pop(rdi);
#endif

        uni_vzeroupper();
        ret();
        this->ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(
                    this->getCode()));
    }
};

template <cpu_isa_t isa, typename conv_t>
inline void init_rtus_driver(conv_t *self) {
    const auto &conf = self->conf_;
    const auto &cd = *conf.cdesc();
    const bool is_bwd_data = cd.prop_kind == prop_kind::backward_data;
    const int ndims = conf.ndims();

    if (!conf.rtus_.reduce_src_) return;

    const int max_threads = mkldnn_get_max_threads();
    size_t factor = 0;
    switch (cd.prop_kind) {
    case prop_kind::forward_training: case prop_kind::forward_inference:
        factor = conf.jcp_.nb_reduce; break;
    case prop_kind::backward_data:
        factor = conf.jcp_.nb_load_blocking_max; break;
    case prop_kind::backward_weights:
        factor = conf.jcp_.nb_bcast_blocking; break;
    default: assert(!"unsupported prop_kind");
    }

    size_t typesize = sizeof(decltype(*self->scratch_));

    self->ws_per_thread_ = factor * conf.jcp_.is * conf.jcp_.ic_block;
    self->scratch_ = (decltype(self->scratch_))malloc(
            max_threads * self->ws_per_thread_ * typesize, 64);

    const int stride_h = (conf.ndims() == 3) ? 1 : cd.strides[0];
    const int stride_w = cd.strides[ndims - 3];

    const auto &src_d = is_bwd_data ? *conf.diff_src_pd()->desc()
                                    : *conf.src_pd()->desc();
    assert((isa == avx2 && utils::one_of(src_d.format, memory_format::nCw8c,
        memory_format::nChw8c)) || (isa == avx512_common && utils::one_of(
            src_d.format, memory_format::nCw16c, memory_format::nChw16c)));

    const int ih = (ndims == 3) ? 1 : src_d.dims[2];
    const int iw = src_d.dims[ndims - 1];

    const int src_step_h = stride_h * iw;
    const int src_step_icb = ih * iw;
    const int ws_step_icb = conf.jcp_.is;
    const bool src_to_ws = !is_bwd_data;
    self->rtus_driver_ = new rtus_driver_t<isa>(iw, stride_w, src_step_h,
            src_step_icb, ws_step_icb, src_to_ws, typesize);
}

inline float loss_ratio(int amount, int divider)
{
    return float(rnd_up(amount, divider) - amount) / rnd_up(amount, divider);
}

inline int best_divider(int value, int min_divider, int max_divider,
                        bool find_max, int step = 1)
{
    max_divider = nstl::max(1, nstl::min(max_divider, value));
    min_divider = nstl::max(1, nstl::min(min_divider, max_divider));

    float min_loss = FLT_MAX;
    int x_divider = max_divider;
    for (int divider = max_divider; divider >= min_divider; divider -= step) {
        const float loss = loss_ratio(value, divider);
        if ((find_max && loss < min_loss) || (!find_max && loss <= min_loss)) {
            min_loss = loss;
            x_divider = divider;
        }
    }
    return x_divider;
}

}
}
}

#endif
