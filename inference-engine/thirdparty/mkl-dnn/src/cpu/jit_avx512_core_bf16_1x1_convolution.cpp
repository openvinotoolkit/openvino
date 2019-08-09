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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "jit_avx512_core_bf16_1x1_convolution.hpp"
#include "utils.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"

#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

#define data_blk_off(f, n, c, h, w) \
    ((ndims == 3) \
    ? (f).blk_off(n, c, w) \
    : (f).blk_off(n, c, h, w))

namespace {
template <typename T, typename U>
void balance2D(U nthr, U ithr, T ny, T &ny_start, T &ny_end,
    T nx, T &nx_start, T &nx_end, T nx_divider)
{
    const T grp_size = utils::div_up(nthr, nx_divider);
    const T grp_count = utils::div_up(nthr, grp_size);

    T grp = ithr / grp_size;
    T grp_ithr = ithr % grp_size;
    T grp_nthr = grp_size;
    T first_grps = nthr % grp_count;
    if (first_grps > 0 && grp >= first_grps) {
        ithr -= first_grps * grp_size;
        grp_nthr--;
        grp = ithr / grp_nthr + first_grps;
        grp_ithr = ithr % grp_nthr;
    }
    balance211(nx, grp_count, grp, nx_start, nx_end);
    balance211(ny, grp_nthr, grp_ithr, ny_start, ny_end);
}
}
/* convolution forward */

template <data_type_t dst_type>
void _jit_avx512_core_bf16_1x1_convolution_fwd_t<dst_type>::execute_forward()
const {
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights =
        reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const float *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    auto scratchpad = this->scratchpad();

    const auto &jcp = kernel_->jcp;
    if (pd()->wants_padded_bias()) {
        auto padded_bias = scratchpad.template get<float>(
                memory_tracking::names::key_conv_padded_bias);
        utils::array_copy(padded_bias, bias, jcp.oc_without_padding);
        utils::array_set(padded_bias + jcp.oc_without_padding, 0.f,
                jcp.oc - jcp.oc_without_padding);
        bias = padded_bias;
    }

    parallel(0, [&](const int ithr, const int nthr) {
        execute_forward_thr(ithr, nthr, src, weights, bias, dst, scratchpad);
    });

    if (pd()->wants_zero_pad_dst())
        output_memory_primitive(0)->zero_pad();
}

template <data_type_t dst_type>
void _jit_avx512_core_bf16_1x1_convolution_fwd_t<dst_type>::execute_forward_thr(
            const int ithr, const int nthr,
            const src_data_t *src, const wei_data_t *weights,
            const float *bias, dst_data_t *dst,
            const memory_tracking::grantor_t &scratchpad)
const {
    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));

    const int ndims = src_d.ndims();
    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[0];
    const int stride_w = pd()->desc()->strides[ndims - 3];
    const int pad_t = (ndims == 3) ? 0 : pd()->desc()->padding[0][0];
    const int pad_l = pd()->desc()->padding[0][ndims - 3];

    const auto &jcp = kernel_->jcp;
    auto rtus_space = scratchpad.template get<src_data_t>(
            memory_tracking::names::key_conv_rtus_space);
    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto p = jit_1x1_conv_call_s();

    auto rp = rtus_driver_t<avx512_common>::call_params_t();
    const int nb_oc = jcp.nb_load;
    const int nb_ic = jcp.nb_reduce;
    const int os_block = jcp.bcast_block;

    int bcast_start{0}, bcast_end{0}, ocb_start{0}, ocb_end{0};
    balance2D(nthr, ithr, work_amount, bcast_start, bcast_end,
        jcp.nb_load, ocb_start, ocb_end, jcp.load_grp_count);

    auto init_bcast = [&](int iwork, int &n, int &g, int &bcast_step,
            int &oh, int &ow, int &ih, int &iw)
    {
        int osb{0};
        nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb,
            jcp.nb_bcast);
        bcast_step = step(jcp.nb_bcast_blocking, jcp.nb_bcast - osb,
                jcp.nb_bcast_blocking_max);
        bcast_step = nstl::min(bcast_step, bcast_end - iwork);

        const int os = osb * os_block;
        oh = os / jcp.ow;
        ow = os % jcp.ow;

        ih = nstl::max(oh * stride_h - pad_t, 0);
        iw = nstl::max(ow * stride_w - pad_l, 0);
        rp.iw_start = iw;

        p.bcast_dim = this_block_size(os, jcp.os,
            bcast_step * os_block);
        rp.os = p.bcast_dim;
    };

    auto init_load = [&](int ocb, int &load_step)
    {
        load_step = step(jcp.nb_load_blocking, ocb_end - ocb,
            jcp.nb_load_blocking_max);
        p.load_dim = this_block_size(ocb * jcp.oc_block,
            ocb_end * jcp.oc_block, load_step * jcp.oc_block);
    };

    auto init_reduce = [&]()
    {
        p.reduce_dim = this_block_size(0, jcp.ic, jcp.ic);
        rp.icb = p.reduce_dim / jcp.reduce_block;
    };

    auto inner_ker = [&](int ocb, int n, int g, int oh, int ow,
        int ih, int iw)
    {
        const int icb = 0;
        const int _ocb = g * nb_oc + ocb;
        const size_t dst_off = data_blk_off(dst_d, n, _ocb , oh, ow);

        p.output_data = &dst[dst_off];
        p.bias_data = &bias[_ocb * jcp.oc_block];
        p.load_data = &weights[pd()->with_groups()
            ? weights_d.blk_off(g, ocb, icb)
            : weights_d.blk_off(ocb, icb)];

        const int _icb = g * nb_ic + icb;
        if (pd()->rtus_.reduce_src_) {
            rp.ws = rtus_space + ithr * pd()->rtus_.space_per_thread_
                + _icb * jcp.is * jcp.ic_block;
            if (ocb == ocb_start) {
                rp.src = src + data_blk_off(src_d, n, _icb, ih, iw);
                rtus_driver_->ker_(&rp);
            }
            p.bcast_data = rp.ws;
        } else
            p.bcast_data = src + data_blk_off(src_d, n, _icb, ih, iw);

        kernel_->jit_ker(&p);
    };

    if (jcp.loop_order == loop_rlb) {
        init_reduce();
        int ocb = ocb_start;
        while (ocb < ocb_end) {
            int load_step;
            init_load(ocb, load_step);
            int iwork = bcast_start;
            while (iwork < bcast_end) {
                int n, g, bcast_step, oh, ow, ih, iw;
                init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
                inner_ker(ocb, n, g, oh, ow, ih, iw);
                iwork += bcast_step;
            }
            ocb += load_step;
        }
    } else if (jcp.loop_order == loop_lbr) {
        int ocb = ocb_start;
        while (ocb < ocb_end) {
            int load_step;
            init_load(ocb, load_step);
            int iwork = bcast_start;
            while (iwork < bcast_end) {
                int n, g, bcast_step, oh, ow, ih, iw;
                init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
                init_reduce();
                inner_ker(ocb, n, g, oh, ow, ih, iw);
                iwork += bcast_step;
            }
            ocb += load_step;
        }
    } else if (jcp.loop_order == loop_rbl) {
        init_reduce();
        int iwork = bcast_start;
        while (iwork < bcast_end) {
            int n, g, bcast_step, oh, ow, ih, iw;
            init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
            int ocb = ocb_start;
            while (ocb < ocb_end) {
                int load_step;
                init_load(ocb, load_step);
                inner_ker(ocb, n, g, oh, ow, ih, iw);
                ocb += load_step;
            }
            iwork += bcast_step;
        }
    } else if (jcp.loop_order == loop_blr) {
        int iwork = bcast_start;
        while (iwork < bcast_end) {
            int n, g, bcast_step, oh, ow, ih, iw;
            init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
            int ocb = ocb_start;
            while (ocb < ocb_end) {
                int load_step;
                init_load(ocb, load_step);
                init_reduce();
                inner_ker(ocb, n, g, oh, ow, ih, iw);
                ocb += load_step;
            }
            iwork += bcast_step;
        }
    } else {
        assert(!"unsupported loop order");
    }
}


template struct _jit_avx512_core_bf16_1x1_convolution_fwd_t<data_type::f32>;
template struct _jit_avx512_core_bf16_1x1_convolution_fwd_t<data_type::bf16>;

template <data_type_t diff_src_type>
void _jit_avx512_core_bf16_1x1_convolution_bwd_data_t<diff_src_type>::execute_backward_data()
const {
    auto diff_dst = reinterpret_cast<const diff_dst_data_t *>
        (this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>
        (this->input_memory(1));
    auto diff_src = reinterpret_cast<diff_src_data_t *>(this->memory());
    auto scratchpad = this->scratchpad();
    parallel(0, [&](const int ithr, const int nthr) {
        execute_backward_data_thr(ithr, nthr, diff_dst, weights, diff_src,
            scratchpad);
    });
}

template <data_type_t diff_src_type>
void _jit_avx512_core_bf16_1x1_convolution_bwd_data_t<diff_src_type>::execute_backward_data_thr(
        const int ithr, const int nthr,
        const diff_dst_data_t *diff_dst, const wei_data_t *weights,
        diff_src_data_t *diff_src,
        const memory_tracking::grantor_t &scratchpad)
const {

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));
    const memory_desc_wrapper diff_src_d(pd()->diff_src_pd());
    const int ndims = diff_src_d.ndims();
    const auto &jcp = kernel_->jcp;
    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    auto rtus_space = scratchpad.template get<diff_src_data_t>(
            memory_tracking::names::key_conv_rtus_space);

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto p = jit_1x1_conv_call_s();

    auto rp = rtus_driver_t<avx512_common>::call_params_t();
    const int nb_ic = jcp.nb_load;
    const int nb_oc = jcp.nb_reduce;
    const int os_block = jcp.bcast_block;

    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[0];
    const int stride_w = pd()->desc()->strides[ndims - 3];
    const int pad_t = (ndims == 3) ? 0 : pd()->desc()->padding[0][0];
    const int pad_l = pd()->desc()->padding[0][ndims - 3];

    int bcast_start{0}, bcast_end{0}, icb_start{0}, icb_end{0};
    balance2D(nthr, ithr, work_amount, bcast_start, bcast_end,
        jcp.nb_load, icb_start, icb_end, jcp.load_grp_count);

    auto init_bcast = [&](const int icb, int iwork, int &n, int &g, int &bcast_step,
            int &oh, int &ow, int &ih, int &iw)
    {
        int osb{0};
        nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb,
            jcp.nb_bcast);
        bcast_step = step(jcp.nb_bcast_blocking, jcp.nb_bcast - osb,
                jcp.nb_bcast_blocking_max);
        bcast_step = nstl::min(bcast_step, bcast_end - iwork);

        const int os = osb * os_block;
        p.bcast_dim = this_block_size(os, jcp.os,
            bcast_step * os_block);
        rp.os = p.bcast_dim;

        oh = os / jcp.ow;
        ow = os % jcp.ow;
        ih = nstl::max(oh * stride_h - pad_t, 0);
        iw = nstl::max(ow * stride_w - pad_l, 0);
        rp.iw_start = iw;
    };

    auto init_load = [&](int icb, int &load_step)
    {
        load_step = step(jcp.nb_load_blocking, icb_end - icb,
            jcp.nb_load_blocking_max);
        p.load_dim = this_block_size(icb * jcp.ic_block,
            icb_end * jcp.ic_block, load_step * jcp.ic_block);
        rp.icb = p.load_dim / jcp.ic_block;
    };

    auto init_reduce = [&]()
    {
        p.reduce_dim = this_block_size(0, jcp.oc, jcp.oc);
    };

    auto inner_ker = [&](int icb, int n, int g, int oh, int ow,
        int ih, int iw)
    {
        const int ocb = 0;
        const int _icb = g * nb_ic + icb;
        const size_t diff_src_off = data_blk_off(diff_src_d, n, _icb , ih, iw);

        rp.src = diff_src + diff_src_off;
        if (pd()->rtus_.reduce_src_) {
            rp.ws = rtus_space + ithr * pd()->rtus_.space_per_thread_;
            p.output_data = rp.ws;
        } else
            p.output_data = rp.src;
        p.load_data = &weights[pd()->with_groups()
            ? weights_d.blk_off(g, ocb, icb)
            : weights_d.blk_off(ocb, icb)];

        const int _ocb = g * nb_oc + ocb;
        p.bcast_data = diff_dst + data_blk_off(diff_dst_d, n, _ocb, oh, ow);

        kernel_->jit_ker(&p);
        if (pd()->rtus_.reduce_src_)
            rtus_driver_->ker_(&rp);
    };

    if (jcp.loop_order == loop_rlb) {
        init_reduce();
        int icb = icb_start;
        while (icb < icb_end) {
            int load_step;
            init_load(icb, load_step);
            int iwork = bcast_start;
            while (iwork < bcast_end) {
                int n{0}, g{0}, bcast_step, oh, ow, ih, iw;
                init_bcast(0, iwork, n, g, bcast_step, oh, ow, ih, iw);
                inner_ker(icb, n, g, oh, ow, ih, iw);
                iwork += bcast_step;
            }
            icb += load_step;
        }
        //XXX: this is the loop order for strided
    } else if (jcp.loop_order == loop_lbr) {
        int icb = icb_start;
        while (icb < icb_end) {
            int load_step;
            init_load(icb, load_step);
            int iwork = bcast_start;
            while (iwork < bcast_end) {
                int n, g, bcast_step, oh, ow, ih, iw;
                init_bcast(icb, iwork, n, g, bcast_step, oh, ow, ih, iw);
                init_reduce();
                inner_ker(icb, n, g, oh, ow, ih, iw);
                iwork += bcast_step;
            }
            icb += load_step;
        }
    } else if (jcp.loop_order == loop_rbl) {
        init_reduce();
        int iwork = bcast_start;
        while (iwork < bcast_end) {
            int n, g, bcast_step, oh, ow, ih, iw;
            init_bcast(0, iwork, n, g, bcast_step, oh, ow, ih, iw);
            int icb = icb_start;
            while (icb < icb_end) {
                int load_step;
                init_load(icb, load_step);
                inner_ker(icb, n, g, oh, ow, ih, iw);
                icb += load_step;
            }
            iwork += bcast_step;
        }
    } else if (jcp.loop_order == loop_blr) {
        int iwork = bcast_start;
        while (iwork < bcast_end) {
            int n, g, bcast_step, oh, ow, ih, iw;
            init_bcast(0, iwork, n, g, bcast_step, oh, ow, ih, iw);
            int icb = icb_start;
            while (icb < icb_end) {
                int load_step;
                init_load(icb, load_step);
                init_reduce();
                inner_ker(icb, n, g, oh, ow, ih, iw);
                icb += load_step;
            }
            iwork += bcast_step;
        }
    } else {
        assert(!"unsupported loop order");
    }

}

template struct _jit_avx512_core_bf16_1x1_convolution_bwd_data_t<data_type::f32>;
template struct _jit_avx512_core_bf16_1x1_convolution_bwd_data_t<data_type::bf16>;

/* convolution backward wtr weights */

#define wht_blk_off(d, g, ...) \
        (pd()->with_groups() \
         ? (d).blk_off((g), __VA_ARGS__) \
         : (d).blk_off(__VA_ARGS__))

template <data_type_t diff_weights_type>
_jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<diff_weights_type>::
        _jit_avx512_core_bf16_1x1_convolution_bwd_weights_t(const pd_t *apd,
                const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(apd, inputs, outputs)
    , kernel_(nullptr), acc_ker_(nullptr), reducer_bias_(nullptr)
    , rtus_driver_(nullptr)
#ifndef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    , tr_reorder_(nullptr)
#endif
{
    kernel_ = new jit_avx512_core_bf16_1x1_conv_kernel(pd()->jcp_, *pd()->attr());

    reducer_bias_ = new cpu_reducer_t<data_type::f32>(pd()->reducer_bia_conf_);
    init_rtus_driver<avx512_common>(this);

    acc_ker_ = new cpu_accumulator_1d_t<data_type::f32>();

#ifndef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    tr_reorder_ = new jit_avx512_core_bf16_reorder_s16c_to_S16c2s_t();
#endif
}

template <data_type_t diff_weights_type>
void _jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<diff_weights_type>::
    execute_backward_weights() const
{
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto diff_dst =
        reinterpret_cast<const diff_dst_data_t *>(this->input_memory(1));
    auto diff_weights = reinterpret_cast<diff_wei_data_t *>(this->memory(0));
    auto diff_bias_in = reinterpret_cast<float *>(this->memory(1));

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_pd());
    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_pd(0));

    const auto &jcp = kernel_->jcp;

    const auto scratchpad = this->scratchpad();

    auto rtus_space = scratchpad.template get<src_data_t>(key_conv_rtus_space);
    float *diff_bias = pd()->wants_padded_bias()
        ? scratchpad.template get<float>(key_conv_padded_bias) : diff_bias_in;
    auto wei_reduction = scratchpad.template get<float>(key_conv_wei_reduction);

#ifndef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    auto tr_src_buffer = scratchpad.template get<src_data_t>(key_conv_tr_src);
    auto tr_diff_buffer =
        scratchpad.template get<diff_dst_data_t>(key_conv_tr_diff_dst);
#endif
    auto d_dst_f32_buffer = scratchpad.template get<float>(key_conv_dst_bf16_convert_wsp);

    const int ndims = src_d.ndims();
    const int wei_size = jcp.ngroups * jcp.oc * jcp.ic;

    simple_barrier::ctx_t reduction_barrier;
    simple_barrier::ctx_init(&reduction_barrier);

    const auto reducer_bia_scratchpad = memory_tracking::grantor_t(scratchpad,
            prefix_reducer_bia);
    auto rb = this->reducer_bias_;
    rb->init(reducer_bia_scratchpad);

    // TODO (Roma): remove this restriction
    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

    const int nb_ic = jcp.nb_bcast;
    const int nb_ic_blocking = jcp.nb_bcast_blocking;

    const int nb_oc = jcp.nb_load;
    const int nb_oc_blocking = jcp.nb_load_blocking;

    const int sp_nb = jcp.nb_reduce;
    const int mb_sp_work = jcp.mb * sp_nb;

    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[0];
    const int stride_w = pd()->desc()->strides[ndims - 3];
    const int pad_t = (ndims == 3) ? 0 : pd()->desc()->padding[0][0];
    const int pad_l = pd()->desc()->padding[0][ndims - 3];
    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto ker = [&](const int ithr, const int nthr) {
        assert(nthr == jcp.nthr);
        assert(IMPLICATION(!mkldnn_thr_syncable(), jcp.nthr_mb == 1));

        const int ithr_ic_b = ithr % jcp.nthr_ic_b;
        const int ithr_oc_b = ithr / jcp.nthr_ic_b % jcp.nthr_oc_b;
        const int ithr_g = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b % jcp.nthr_g;
        const int ithr_mb = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b /
                            jcp.nthr_g;

        /* reduction dimension */
        int mb_sp_b_start{ 0 }, mb_sp_b_end{ 0 };
        balance211(mb_sp_work, jcp.nthr_mb, ithr_mb, mb_sp_b_start,
                mb_sp_b_end);

        /* independent dimensions */
        int g_start{ 0 }, oc_b_start{ 0 }, ic_b_start{ 0 };
        int g_end{ 0 }, oc_b_end{ 0 }, ic_b_end{ 0 };

        balance211(jcp.ngroups, jcp.nthr_g, ithr_g, g_start, g_end);
        balance211(jcp.nb_load, jcp.nthr_oc_b, ithr_oc_b, oc_b_start,
                    oc_b_end);
        balance211(jcp.nb_bcast, jcp.nthr_ic_b, ithr_ic_b, ic_b_start,
                    ic_b_end);

        const int g_work = g_end - g_start;
        const int oc_b_work = oc_b_end - oc_b_start;
        const int ic_b_work = ic_b_end - ic_b_start;

        float *diff_wei;
        if (diff_weights_type == data_type::bf16) {
            diff_wei = wei_reduction + (ithr_mb) * wei_size;
        } else {
            diff_wei = ithr_mb == 0 ?
                    (float*)diff_weights :
                    (float*)wei_reduction + (ithr_mb - 1) * wei_size;
        }

        int sp_b_step = 0;
        for (int mb_sp_b = mb_sp_b_start; mb_sp_b < mb_sp_b_end;
                mb_sp_b += sp_b_step) {
            int img{ 0 }, sp_b{ 0 };
            nd_iterator_init(mb_sp_b, img, jcp.mb, sp_b, sp_nb);
            sp_b_step = step(jcp.nb_reduce_blocking,
                    nstl::min(sp_nb - sp_b, mb_sp_b_end - mb_sp_b),
                    jcp.nb_reduce_blocking_max);

            for (int g = g_start; g < g_end; ++g) {
                int load_step = 0;
                int bcast_step = 0;
                for (int ic_b = ic_b_start; ic_b < ic_b_end;
                        ic_b += bcast_step) {
                    bcast_step = step(nb_ic_blocking, ic_b_end - ic_b,
                            jcp.nb_bcast_blocking_max);
                    for (int oc_b = oc_b_start; oc_b < oc_b_end;
                            oc_b += load_step) {
                        load_step = step(nb_oc_blocking, oc_b_end - oc_b,
                                jcp.nb_load_blocking_max);
                        const int _ic_b = g * nb_ic + ic_b;
                        const int _oc_b = g * nb_oc + oc_b;

                        float *store_to;

                        const size_t off
                                = wht_blk_off(diff_weights_d, g, oc_b, ic_b);
                        store_to = diff_wei + off;

                        const src_data_t *diff_src =
                                &src[src_d.blk_off(img, _ic_b)];

                        int sp_b_end = sp_b + sp_b_step;
                        const diff_dst_data_t *pdiff_dst
                                = &diff_dst[diff_dst_d.blk_off(img, _oc_b)];
                        const src_data_t *local_src = diff_src;

                        auto p = jit_1x1_conv_call_s();
                        auto rp = rtus_driver_t<avx512_common>::call_params_t();

                        p.output_stride
                                = jcp.ic * jcp.oc_block * jcp.typesize_out;

                        p.load_dim = load_step * jcp.oc_block;

                        p.bcast_dim = bcast_step * jcp.ic_block;
                        rp.icb = bcast_step;
                        p.output_data = store_to;

#ifndef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
                        p.reduce_dim = nstl::min(sp_b_step * jcp.reduce_block,
                                  jcp.reduce_dim - sp_b * jcp.reduce_block);
#else
                        p.reduce_dim = sp_b_step * jcp.reduce_block;
#endif
                        rp.os = p.reduce_dim;

                        p.first_last_flag = 0
                            | (mb_sp_b == mb_sp_b_start ? FLAG_REDUCE_FIRST : 0)
                            | (sp_b_end == sp_nb ? FLAG_SP_LAST : 0);

                        int sp = sp_b * jcp.reduce_block;
                        p.load_data = pdiff_dst + sp * jcp.oc_block;

                        if (pd()->rtus_.reduce_src_) {
                            const int oh = sp / jcp.ow;
                            const int ow = sp % jcp.ow;

                            const int ih = nstl::max(oh * stride_h - pad_t, 0);
                            const int iw = nstl::max(ow * stride_w - pad_l, 0);
                            rp.iw_start = iw;

                            rp.ws = rtus_space
                                + ithr * pd()->rtus_.space_per_thread_
                                + sp * jcp.ic_block;

                            if (ndims == 3)
                                rp.src = local_src + iw
                                    * src_d.blocking_desc().strides[0][2];
                            else
                                rp.src = local_src + ih
                                    * src_d.blocking_desc().strides[0][2]
                                    + iw * src_d.blocking_desc().strides[0][3];
                            rtus_driver_->ker_(&rp);

                            p.bcast_data = rp.ws;
                        } else
                            p.bcast_data = local_src + sp * jcp.ic_block;
#ifndef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
                        bf16_cvt_utils::jit_call_t ptr;
                        ptr.size = p.reduce_dim;
                        int thr_src_block_size = rnd_up(jcp.reduce_dim, 2)
                            * jcp.ic_block * jcp.nb_bcast_blocking_max;
                        src_data_t *tr_src =
                            &tr_src_buffer[ithr * thr_src_block_size];
                        for (int bs = 0; bs < bcast_step; bs++) {
                            size_t src_off =
                                bs * jcp.reduce_dim * jcp.ic_block;
                            size_t src_tr_off =
                                bs * rnd_up(jcp.reduce_dim, 2) * jcp.ic_block;
                            src_data_t *curr_inp =
                                &((src_data_t *)p.bcast_data)[src_off];
                            src_data_t *curr_out = &tr_src[src_tr_off];
                            ptr.inp = (void *)curr_inp;
                            ptr.out = (void *)curr_out;
                            tr_reorder_->jit_ker(&ptr);
                        }

                        p.bcast_data = (void *)tr_src;

                        int thr_dst_block_size = rnd_up(jcp.reduce_dim, 2)
                            * jcp.oc_block * jcp.nb_load_blocking_max;
                        diff_dst_data_t *tr_diff_dst =
                            &tr_diff_buffer[ithr * thr_dst_block_size];
                        for (int ls = 0; ls < load_step; ls++) {
                            size_t ddst_off = ls * jcp.os * jcp.oc_block;
                            size_t ddst_tr_off =
                                ls * rnd_up(jcp.reduce_dim, 2)* jcp.oc_block;
                            diff_dst_data_t *curr_inp =
                                &((diff_dst_data_t *)p.load_data)[ddst_off];
                            diff_dst_data_t *curr_out =
                                &tr_diff_dst[ddst_tr_off];
                            ptr.inp = (void *)curr_inp;
                            ptr.out = (void *)curr_out;
                            tr_reorder_->jit_ker(&ptr);
                        }
                        p.load_data = (void *)tr_diff_dst;

#endif //BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
                        kernel_->jit_ker(&p);
                    }
                }
            }
        }

        const int _start_nthr_mb = 1;
        const bool is_bf16_out = diff_weights_type == data_type::bf16;
        /* diff_weights[:] += sum(ws_reduction_[thr_mb][:]) */
        if (jcp.nthr_mb > _start_nthr_mb) {
            simple_barrier::barrier(&reduction_barrier, jcp.nthr);
            const int work = g_work * oc_b_work * ic_b_work;
            int start{ 0 }, end{ 0 };
            balance211(work, jcp.nthr_mb, ithr_mb, start, end);
            if (start == end)
                return;

            for (int thr_mb = _start_nthr_mb; thr_mb < jcp.nthr_mb; ++thr_mb) {
                int w = start;
                int sub_g_start{ 0 }, sub_oc_b_start{ 0 },
                        sub_ic_b_start{ 0 };
                nd_iterator_init(w, sub_g_start, g_work, sub_oc_b_start,
                        oc_b_work, sub_ic_b_start, ic_b_work);
                while (w < end) {
                    const int g = g_start + sub_g_start;
                    const int oc_b = oc_b_start + sub_oc_b_start;
                    const int ic_b = ic_b_start + sub_ic_b_start;

                    const size_t acc_size
                            = (size_t)jcp.ic_block * jcp.oc_block
                            * nstl::min(end - w, ic_b_work - sub_ic_b_start);

                    const size_t off
                            = wht_blk_off(diff_weights_d, g, oc_b, ic_b);
                    float *wei_reduced = is_bf16_out
                        ? wei_reduction + off
                        : (float*)diff_weights + off;

                    int thr_mb_buffer_idx = is_bf16_out ? thr_mb : thr_mb - 1;
                    float *wei_to_reduce = wei_reduction
                        + thr_mb_buffer_idx * wei_size + off;
                    if (is_bf16_out && thr_mb == jcp.nthr_mb - 1)
                        // the last iteration for bfloat16 requires conversion
                        // and store to diff_weights array
                        bf16_cvt_utils::add_floats_and_cvt_to_bfloat16(
                            (mkldnn_bfloat16_t *)(diff_weights + off),
                            wei_reduced, wei_to_reduce, acc_size);
                    else
                        acc_ker_->accumulate(
                            wei_reduced, wei_to_reduce, acc_size);

                    nd_iterator_jump(w, end, sub_g_start, g_work,
                            sub_oc_b_start, oc_b_work, sub_ic_b_start,
                            ic_b_work);
                }
            }
        } else if (is_bf16_out) {
            for (int g = g_start; g < g_end; g++)
            for (int oc_b = oc_b_start; oc_b < oc_b_end; oc_b++) {
                const size_t acc_size = (size_t)ic_b_work
                    * jcp.ic_block * jcp.oc_block;
                const size_t off =
                    wht_blk_off(diff_weights_d, g, oc_b, ic_b_start);

                bf16_cvt_utils::cvt_float_to_bfloat16(
                    (mkldnn_bfloat16_t *)(diff_weights + off),
                    (const float*)(wei_reduction + off), acc_size);
            }
        }
    };

    auto ker_bias = [&](int ithr, int nthr) {
        assert(nthr == rb->balancer().nthr_);

        const int batch_job_start = rb->balancer().ithr_job_off(ithr);
        const int batch_njobs = rb->balancer().ithr_njobs(ithr);

        if (batch_njobs == 0)
            return;

        /* reduction dimension */
        int img_start{ 0 }, img_end{ 0 };

        balance211(jcp.mb, rb->balancer().nthr_per_group_,
                rb->balancer().id_in_group(ithr), img_start, img_end);

        /* jobs */
        int g_start{ 0 }, ocb_start{ 0 };
        nd_iterator_init(
                batch_job_start, g_start, jcp.ngroups, ocb_start, jcp.nb_load);

        for (int img = img_start; img < img_end; ++img) {
            int g = g_start, ocb = ocb_start;
            for (int batch_job_loc = 0; batch_job_loc < batch_njobs; ++batch_job_loc) {
                const size_t _oc = g * jcp.nb_load + ocb;

                const diff_dst_data_t *d_dst = &diff_dst[diff_dst_d.blk_off(img, _oc)];
                float *d_bias = &rb->get_local_ptr(ithr, diff_bias,
                        reducer_bia_scratchpad)[batch_job_loc * rb->balancer().job_size_];

                const size_t d_dst_f32_size = (size_t)jcp.oh * jcp.ow * jcp.oc_block;
                auto dst_ws = d_dst_f32_buffer + d_dst_f32_size * ithr;

                bf16_cvt_utils::cvt_bfloat16_to_float(dst_ws, d_dst, d_dst_f32_size);

                if (img == img_start)
                    for (int o = 0; o < 16; ++o)
                        d_bias[o] = 0.;

                for (int hw = 0; hw < jcp.oh * jcp.ow; ++hw) {
                    PRAGMA_OMP_SIMD()
                    for (int o = 0; o < 16; ++o)
                        d_bias[o] += dst_ws[o];
                    dst_ws += 16;
                }

                nd_iterator_step(g, jcp.ngroups, ocb, jcp.nb_load);
            }
        }
        rb->reduce(ithr, diff_bias, reducer_bia_scratchpad);
    };

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        ker(ithr, jcp.nthr);
        if (pd()->with_bias())
            ker_bias(ithr, jcp.nthr);
    });

    /* TODO: put this in ker_bias */
    if (pd()->wants_padded_bias()) {
        assert(jcp.ngroups == 1);
        utils::array_copy(diff_bias_in, diff_bias, jcp.oc_without_padding);
    }
}

template struct _jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<data_type::f32>;
template struct _jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<data_type::bf16>;

}
}
}
