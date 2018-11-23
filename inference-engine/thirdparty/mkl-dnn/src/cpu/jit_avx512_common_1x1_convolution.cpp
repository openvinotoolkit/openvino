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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "jit_avx512_common_1x1_convolution.hpp"
#include "utils.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"

#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

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

template <bool with_relu, data_type_t src_type, data_type_t wei_type,
        data_type_t dst_type>
void _jit_avx512_common_1x1_convolution_fwd_t
    <with_relu, src_type, wei_type, dst_type>::execute_forward()
{
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights =
        reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const dst_data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    auto &jcp = kernel_->jcp;
    if (conf_.want_padded_bias()) {
        assert(jcp.ngroups == 1);
        for (int oc = 0; oc < jcp.oc_without_padding; ++oc)
            padded_bias_[oc] = bias[oc];
        bias = padded_bias_;
    }

    parallel(0, [&](const int ithr, const int nthr) {
        execute_forward_thr(ithr, nthr, src, weights, bias, dst);
    });
}

template <bool with_relu, data_type_t src_type, data_type_t wei_type,
        data_type_t dst_type>
void _jit_avx512_common_1x1_convolution_fwd_t
    <with_relu, src_type, wei_type, dst_type>::execute_forward_thr(
            const int ithr, const int nthr,
            const src_data_t *src, const wei_data_t *weights,
            const dst_data_t *bias, dst_data_t *dst)
{
    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));

    const int stride_h = conf_.cdesc()->strides[0];
    const int stride_w = conf_.cdesc()->strides[1];
    const int pad_t = conf_.cdesc()->padding[0][0];
    const int pad_l = conf_.cdesc()->padding[0][1];

    auto &jcp = kernel_->jcp;
    const int MB = conf_.MB();
    const int work_amount = MB * jcp.ngroups * jcp.nb_bcast;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto p = jit_1x1_conv_call_s();

    auto rp = rtus_driver_t<avx512_common>::call_params_t();

    const int nb_oc = jcp.nb_load;
    const int nb_ic = jcp.nb_reduce;
    const int nb_ic_blocking = jcp.nb_reduce_blocking;
    const int os_block = jcp.bcast_block;

    int bcast_start{0}, bcast_end{0}, ocb_start{0}, ocb_end{0};
    balance2D(nthr, ithr, work_amount, bcast_start, bcast_end,
        jcp.nb_load, ocb_start, ocb_end, jcp.load_grp_count);

    auto init_bcast = [&](int iwork, int &n, int &g, int &bcast_step,
            int &oh, int &ow, int &ih, int &iw)
    {
        int osb{0};
        nd_iterator_init(iwork, n, MB, g, jcp.ngroups, osb,
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

    auto init_reduce = [&](int icb)
    {
        const int nb_ic_blocking_step =
            nstl::min(icb + nb_ic_blocking, nb_ic) - icb;
        p.first_last_flag = 0
            | (icb == 0 ? FLAG_REDUCE_FIRST : 0)
            | (icb + nb_ic_blocking_step >= nb_ic
                    ? FLAG_REDUCE_LAST : 0);

        p.reduce_dim = this_block_size(icb * jcp.ic_block,
            jcp.ic, nb_ic_blocking_step * jcp.ic_block);
        rp.icb = p.reduce_dim / jcp.reduce_block;
    };

    auto inner_ker = [&](int ocb, int icb, int n, int g, int oh, int ow,
        int ih, int iw)
    {

        const int _ocb = g * nb_oc + ocb;
        const size_t dst_off = dst_d.blk_off(n, _ocb, oh, ow);

        p.output_data = &dst[dst_off];
        p.bias_data = &bias[_ocb * jcp.oc_block];
        p.load_data = &weights[conf_.with_groups()
            ? weights_d.blk_off(g, ocb, icb)
            : weights_d.blk_off(ocb, icb)];

        const int _icb = g * nb_ic + icb;
        if (conf_.rtus_.reduce_src_) {
            rp.ws = scratch_ + ithr * ws_per_thread_
                + _icb * jcp.is * jcp.ic_block;
            if (ocb == ocb_start) {
                rp.src = src + src_d.blk_off(n, _icb, ih, iw);
                rtus_driver_->ker_(&rp);
            }
            p.bcast_data = rp.ws;
        } else
            p.bcast_data = src + src_d.blk_off(n, _icb, ih, iw);

        p.oc_off = _ocb * jcp.oc_block * sizeof(dst_data_t);

        kernel_->jit_ker(&p);
    };

    if (jcp.loop_order == loop_rlb) {
        for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
            init_reduce(icb);
            int ocb = ocb_start;
            while (ocb < ocb_end) {
                int load_step;
                init_load(ocb, load_step);
                int iwork = bcast_start;
                while (iwork < bcast_end) {
                    int n, g, bcast_step, oh, ow, ih, iw;
                    init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
                    inner_ker(ocb, icb, n, g, oh, ow, ih, iw);
                    iwork += bcast_step;
                }
                ocb += load_step;
            }
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
                for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                    init_reduce(icb);
                    inner_ker(ocb, icb, n, g, oh, ow, ih, iw);
                }
                iwork += bcast_step;
            }
            ocb += load_step;
        }
    } else if (jcp.loop_order == loop_rbl) {
        for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
            init_reduce(icb);
            int iwork = bcast_start;
            while (iwork < bcast_end) {
                int n, g, bcast_step, oh, ow, ih, iw;
                init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
                int ocb = ocb_start;
                while (ocb < ocb_end) {
                    int load_step;
                    init_load(ocb, load_step);
                    inner_ker(ocb, icb, n, g, oh, ow, ih, iw);
                    ocb += load_step;
                }
                iwork += bcast_step;
            }
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
                for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                    init_reduce(icb);
                    inner_ker(ocb, icb, n, g, oh, ow, ih, iw);
                }
                ocb += load_step;
            }
            iwork += bcast_step;
        }
    } else {
        assert(!"unsupported loop order");
    }
}


template struct _jit_avx512_common_1x1_convolution_fwd_t<true, data_type::f32>;
template struct _jit_avx512_common_1x1_convolution_fwd_t<false, data_type::f32>;
template struct _jit_avx512_common_1x1_convolution_fwd_t<false, data_type::s16,
    data_type::s16, data_type::s32>;
template struct _jit_avx512_common_1x1_convolution_fwd_t<true, data_type::s16,
    data_type::s16, data_type::s32>;
/* convolution backward wtr data */

template <data_type_t diff_dst_type, data_type_t wei_type,
    data_type_t diff_src_type>
void _jit_avx512_common_1x1_convolution_bwd_data_t
    <diff_dst_type, wei_type, diff_src_type>::execute_backward_data()
{
    auto diff_dst = reinterpret_cast<const diff_dst_data_t *>
        (this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>
        (this->input_memory(1));
    auto diff_src = reinterpret_cast<diff_src_data_t *>(this->memory());

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper diff_src_d(conf_.diff_src_pd());

    const auto &jcp = kernel_->jcp;
    const int MB = conf_.MB();

    // TODO (Roma): remove this restriction
    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

    const int stride_h = conf_.desc()->strides[0];
    const int stride_w = conf_.desc()->strides[1];
    const int pad_t = conf_.desc()->padding[0][0];
    const int pad_l = conf_.desc()->padding[0][1];

    const int nb_ic = jcp.nb_load;
    const int nb_oc = jcp.nb_reduce;
    const int os_block = jcp.bcast_block;
    const int nb_oc_blocking = jcp.nb_reduce_blocking;

    const int work_amount = MB * jcp.ngroups * jcp.nb_bcast;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    parallel(0, [&](const int ithr, const int nthr) {
        auto p = jit_1x1_conv_call_s();
        auto rp = rtus_driver_t<avx512_common>::call_params_t();

        int bcast_start{0}, bcast_end{0}, icb_start{0}, icb_end{0};
        balance2D(nthr, ithr, work_amount, bcast_start, bcast_end,
            jcp.nb_load, icb_start, icb_end, jcp.load_grp_count);

        bool reduce_outer = (jcp.loop_order == loop_rbl
            || jcp.loop_order == loop_rlb);
        int nboc_outer = reduce_outer ? nb_oc : 1;
        int ocb_outer_step = reduce_outer ? nb_oc_blocking : 1;

        int nboc_inner = reduce_outer ? 1 : nb_oc;
        int ocb_inner_step = reduce_outer ? 1 : nb_oc_blocking;

        for (int ocb_outer = 0; ocb_outer < nboc_outer;
            ocb_outer += ocb_outer_step) {
            size_t cur_ocb_outer =
                nstl::min(ocb_outer + ocb_outer_step, nboc_outer) - ocb_outer;

            int load_step = 0;
            for (int icb = icb_start; icb < icb_end; icb += load_step) {
                load_step = step(jcp.nb_load_blocking, jcp.nb_load - icb,
                        jcp.nb_load_blocking_max);

                p.load_dim = this_block_size(icb * jcp.ic_block,
                    icb_end * jcp.ic_block, load_step * jcp.ic_block);
                rp.icb = p.load_dim / jcp.ic_block;

                int bcast_step;
                for (int iwork = bcast_start; iwork < bcast_end;
                    iwork += bcast_step)
                {
                    int n{0}, g{0}, osb{0};
                    nd_iterator_init(iwork, n, MB, g, jcp.ngroups, osb,
                            jcp.nb_bcast);

                    bcast_step = step(jcp.nb_bcast_blocking, jcp.nb_bcast - osb,
                            jcp.nb_bcast_blocking_max);
                    bcast_step = nstl::min(bcast_step, bcast_end - iwork);

                    const int os = osb * os_block;
                    p.bcast_dim = this_block_size(os, jcp.os,
                            bcast_step * os_block);
                    rp.os = p.bcast_dim;

                    const int oh = os / jcp.ow;
                    const int ow = os % jcp.ow;
                    const int ih = nstl::max(oh * stride_h - pad_t, 0);
                    const int iw = nstl::max(ow * stride_w - pad_l, 0);
                    rp.iw_start = iw;

                    const int _icb = g * nb_ic + icb;
                    rp.src = diff_src + diff_src_d.blk_off(n, _icb, ih, iw);

                    if (conf_.rtus_.reduce_src_) {
                        rp.ws = scratch_ + ithr * ws_per_thread_;
                        p.output_data = rp.ws;
                    } else
                        p.output_data = rp.src;

                    for (int ocb_inner = 0; ocb_inner < nboc_inner;
                        ocb_inner += ocb_inner_step) {
                        int cur_ocb_inner =
                            nstl::min(ocb_inner + ocb_inner_step, nboc_inner) -
                            ocb_inner;

                        int ocb = reduce_outer ? ocb_outer : ocb_inner;
                        int nb_oc_blocking_step = reduce_outer
                            ? cur_ocb_outer : cur_ocb_inner;
                        const int _ocb = g * nb_oc + ocb;
                        size_t diff_dst_off =
                            diff_dst_d.blk_off(n, _ocb, oh, ow);
                        p.bcast_data = &diff_dst[diff_dst_off];

                        p.load_data = &weights[conf_.with_groups()
                            ? weights_d.blk_off(g, ocb, icb)
                            : weights_d.blk_off(ocb, icb)];

                        p.first_last_flag = ocb == 0 ? FLAG_REDUCE_FIRST : 0;

                        p.reduce_dim = this_block_size(ocb * jcp.oc_block,
                            jcp.oc, nb_oc_blocking_step * jcp.oc_block);

                        kernel_->jit_ker(&p);
                    }
                    if (conf_.rtus_.reduce_src_)
                        rtus_driver_->ker_(&rp);
                }
            }
        }
    });
}

template struct _jit_avx512_common_1x1_convolution_bwd_data_t<data_type::f32>;
template struct _jit_avx512_common_1x1_convolution_bwd_data_t<data_type::s16,
    data_type::s16, data_type::s32>;

/* convolution backward wtr weights */

#define wht_blk_off(d, g, ...) \
        (conf_.with_groups() \
         ? (d).blk_off((g), __VA_ARGS__) \
         : (d).blk_off(__VA_ARGS__))

jit_avx512_common_1x1_convolution_bwd_weights_t ::
        jit_avx512_common_1x1_convolution_bwd_weights_t(const pd_t *pd,
                const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs)
    , conf_(*pd), kernel_(nullptr), acc_ker_(nullptr), reducer_bias_(nullptr)
    , trans_kernel_(nullptr), rtus_driver_(nullptr), ws_per_thread_(0)
    , scratch_(nullptr), padded_bias_(nullptr), bctx_(nullptr)
    , tr_src_(nullptr), ws_reduction_(nullptr)
{
    kernel_ = new jit_avx512_common_1x1_conv_kernel(conf_.jcp_, *conf_.attr());

    const auto &jcp = kernel_->jcp;

    const int wei_size = jcp.ngroups * jcp.oc * jcp.ic;
    ws_reduction_ =
        (data_t *)malloc((jcp.nthr_mb - 1) * wei_size * sizeof(data_t), 64);
    acc_ker_ = new cpu_accumulator_1d_t<data_type::f32>();

    if (conf_.with_bias()) {
        const size_t max_buffer_size = jcp.nthr * 3 * 5 * 5 * 16 * 16;
        reducer_bias_ = new cpu_reducer_t<data_type::f32>(
                reduce_balancer_t(jcp.nthr, jcp.oc_block,
                        jcp.ngroups * jcp.nb_load, jcp.mb, max_buffer_size));

        if (conf_.want_padded_bias()) {
            assert(jcp.ngroups == 1);
            padded_bias_ = (data_t *)malloc(sizeof(data_t) * jcp.oc, 64);
        }
    }

    if (jcp.transpose_src) {
        const ptrdiff_t tr_src_size = (ptrdiff_t)jcp.nthr_mb
            * (ptrdiff_t)jcp.ngroups * (ptrdiff_t)jcp.ic * jcp.tr_is;
        tr_src_ = (data_t *)malloc(tr_src_size * sizeof(data_t), 64);
        parallel_nd(tr_src_size, [&](ptrdiff_t i) { tr_src_[i] = 0; });
        auto tp = jit_transpose4x16_src_t();
        tp.src_pf0_distance = 4;
        tp.tr_src_pf0_distance = 0;
        tp.src_pf1 = true;
        tp.tr_src_pf1 = false;
        trans_kernel_ = new jit_transpose4x16_src(&jcp, &tp);

        bctx_ = (simple_barrier::ctx_t *)malloc(
                jcp.nthr * sizeof(simple_barrier::ctx_t), 64);
        for (int i = 0; i < jcp.nthr; ++i)
            simple_barrier::ctx_init(&bctx_[i]);
    }

    init_rtus_driver<avx512_common>(this);
}

void jit_avx512_common_1x1_convolution_bwd_weights_t::execute_backward_weights()
{
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_weights = reinterpret_cast<data_t *>(this->memory(0));
    auto diff_bias_in = reinterpret_cast<data_t *>(this->memory(1));
    data_t *diff_bias = conf_.want_padded_bias() ? padded_bias_ : diff_bias_in;

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper diff_weights_d(conf_.diff_weights_pd(0));

    const auto &jcp = kernel_->jcp;
    const int wei_size = jcp.ngroups * jcp.oc * jcp.ic;

    simple_barrier::ctx_t reduction_barrier;
    simple_barrier::ctx_init(&reduction_barrier);

    // TODO (Roma): remove this restriction
    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

    const int nb_ic = jcp.nb_bcast;
    const int nb_ic_blocking = jcp.nb_bcast_blocking;

    const int nb_oc = jcp.nb_load;
    const int nb_oc_blocking = jcp.nb_load_blocking;

    const int sp_nb = jcp.nb_reduce;
    const int mb_sp_work = jcp.mb * sp_nb;

    const int stride_h = conf_.desc()->strides[0];
    const int stride_w = conf_.desc()->strides[1];
    const int pad_t = conf_.desc()->padding[0][0];
    const int pad_l = conf_.desc()->padding[0][1];

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    // TODO: use memory descriptor with the same fmt as src
    // (or use a macro :))
    auto tr_src_off = [&](int img, int icb, int is) {
        const size_t tr_chn_size = jcp.tr_is * jcp.ic_block;
        const size_t tr_img_size = tr_chn_size * nb_ic * jcp.ngroups;
        return img * tr_img_size + icb * tr_chn_size + is * jcp.ic_block;
    };

    auto uker_trans = [&](int ithr_mb, int img, int sp_b_start, int sp_size,
        int g_start, int g_work, int ic_b_start, int ic_b_work,
        int ithr, int nthr, int first_ic_b)
    {
        const int work_amount = g_work * ic_b_work;

        int start{ 0 }, end{ 0 };
        balance211(work_amount, nthr, ithr, start, end);

        int g{ 0 }, ic_b{ 0 };
        nd_iterator_init(start, g, g_work, ic_b, ic_b_work);
        g += g_start;
        const int ic_b_tr = g * nb_ic + first_ic_b + ic_b;
        ic_b += ic_b_start;

        const int _ic = g * nb_ic + ic_b;

        const int is = sp_b_start * jcp.reduce_block;
        const int ih = is / jcp.iw;
        const int iw = is % jcp.iw;

        data_t *src1 = (data_t *)&src[src_d.blk_off(img, _ic, ih, iw)];
        data_t *tr_src1 = &tr_src_[tr_src_off(ithr_mb, ic_b_tr, is)];

        assert(jcp.ic_block == 16);
        const int src_stride = jcp.is * jcp.ic_block;
        const int tr_src_stride = jcp.tr_is * jcp.ic_block;

        const int my_work = end - start;
        for (int iwork = 0; iwork < my_work; iwork++) {
            auto par_trans = jit_src_transpose_s();
            assert(sp_size % 4 == 0 || sp_size % 4 == jcp.is % 4);
            par_trans.size = sp_size;
            par_trans.src = src1;
            par_trans.tr_src = tr_src1;
            par_trans.src_prf = src1 + 64 * 16;
            par_trans.tr_src_prf = tr_src1 + 80 * 16;
            trans_kernel_->jit_ker(&par_trans);

            src1 += src_stride;
            tr_src1 += tr_src_stride;
        }
    };

    auto ker = [&](const int ithr, const int nthr) {
        assert(nthr == jcp.nthr);
        assert(utils::implication(!mkldnn_thr_syncable(), jcp.nthr_mb == 1));

        const int ithr_ic_b = ithr % jcp.nthr_ic_b;
        const int ithr_oc_b = ithr / jcp.nthr_ic_b % jcp.nthr_oc_b;
        const int ithr_g = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b % jcp.nthr_g;
        const int ithr_mb = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b /
                            jcp.nthr_g;

        const int ithr_but_oc
                = (ithr_mb * jcp.nthr_g + ithr_g) * jcp.nthr_ic_b + ithr_ic_b;

        /* reduction dimension */
        int mb_sp_b_start{ 0 }, mb_sp_b_end{ 0 };
        if (jcp.transpose_src && jcp.nthr_mb < jcp.mb / 2) {
            // it's preferable to parallelize by mb if possible
            int img_start{ 0 }, img_end{ 0 };
            balance211(jcp.mb, jcp.nthr_mb, ithr_mb, img_start, img_end);
            mb_sp_b_start = img_start * sp_nb;
            mb_sp_b_end = img_end * sp_nb;
        }
        else {
            balance211(mb_sp_work, jcp.nthr_mb, ithr_mb, mb_sp_b_start,
                    mb_sp_b_end);
        }

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

        data_t *diff_wei = ithr_mb == 0 ?
                diff_weights :
                ws_reduction_ + (ithr_mb - 1) * wei_size;

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
                    if (jcp.transpose_src) {
                        if (jcp.nthr_oc_b > 1)
                            simple_barrier::barrier(
                                    &bctx_[ithr_but_oc], jcp.nthr_oc_b);
                        const int sp_size
                                = nstl::min(sp_b_step * jcp.reduce_block,
                                        jcp.is - sp_b * jcp.reduce_block);
                        uker_trans(ithr_mb, img, sp_b, sp_size, g, 1, ic_b,
                            bcast_step, ithr_oc_b, jcp.nthr_oc_b, ic_b_start);
                        if (jcp.nthr_oc_b > 1)
                            simple_barrier::barrier(
                                    &bctx_[ithr_but_oc], jcp.nthr_oc_b);
                    }

                    for (int oc_b = oc_b_start; oc_b < oc_b_end;
                            oc_b += load_step) {
                        load_step = step(nb_oc_blocking, oc_b_end - oc_b,
                                jcp.nb_load_blocking_max);
                        const int _ic_b = g * nb_ic + ic_b;
                        const int _ic_b_tr = g * nb_ic + ic_b_start;
                        const int _oc_b = g * nb_oc + oc_b;

                        data_t *store_to;

                        const size_t off
                                = wht_blk_off(diff_weights_d, g, oc_b, ic_b);
                        store_to = diff_wei + off;

                        const data_t *diff_src = jcp.transpose_src ?
                                &tr_src_[tr_src_off(ithr_mb, _ic_b_tr, 0)] :
                                &src[src_d.blk_off(img, _ic_b)];

                        int sp_b_end = sp_b + sp_b_step;
                        const data_t *pdiff_dst
                                = &diff_dst[diff_dst_d.blk_off(img, _oc_b)];
                        const data_t *local_src = diff_src;

                        auto p = jit_1x1_conv_call_s();
                        auto rp = rtus_driver_t<avx512_common>::call_params_t();

                        p.output_stride
                                = jcp.ic * jcp.oc_block * jcp.typesize_out;

                        p.load_dim = load_step * jcp.oc_block;

                        p.bcast_dim = bcast_step * jcp.ic_block;
                        rp.icb = bcast_step;
                        p.output_data = store_to;

                        p.reduce_dim = sp_b_step * jcp.reduce_block;
                        rp.os = p.reduce_dim;

                        p.first_last_flag = 0
                            | (mb_sp_b == mb_sp_b_start ? FLAG_REDUCE_FIRST : 0)
                            | (sp_b_end == sp_nb ? FLAG_SP_LAST : 0);

                        int sp = sp_b * jcp.reduce_block;
                        p.load_data = pdiff_dst + sp * jcp.oc_block;

                        if (conf_.rtus_.reduce_src_) {
                            const int oh = sp / jcp.ow;
                            const int ow = sp % jcp.ow;

                            const int ih = nstl::max(oh * stride_h - pad_t, 0);
                            const int iw = nstl::max(ow * stride_w - pad_l, 0);
                            rp.iw_start = iw;

                            rp.ws = scratch_ + ithr * ws_per_thread_
                                    + sp * jcp.ic_block;
                            rp.src = local_src
                                    + ih * src_d.blocking_desc().strides[0][2]
                                    + iw * src_d.blocking_desc().strides[0][3];

                            rtus_driver_->ker_(&rp);

                            p.bcast_data = rp.ws;
                        } else
                            p.bcast_data = local_src + sp * jcp.ic_block;

                        kernel_->jit_ker(&p);
                    }
                }
            }
        }

        /* diff_weights[:] += sum(ws_reduction_[thr_mb][:]) */
        if (jcp.nthr_mb > 1) {
            simple_barrier::barrier(&reduction_barrier, jcp.nthr);
            const int work = g_work * oc_b_work * ic_b_work;
            int start{ 0 }, end{ 0 };
            balance211(work, jcp.nthr_mb, ithr_mb, start, end);
            if (start == end)
                return;

            for (int thr_mb = 1; thr_mb < jcp.nthr_mb; ++thr_mb) {
                int w = start;
                int sub_g_start{ 0 }, sub_oc_b_start{ 0 },
                        sub_ic_b_start{ 0 };
                nd_iterator_init(w, sub_g_start, g_work, sub_oc_b_start,
                        oc_b_work, sub_ic_b_start, ic_b_work);
                while (w < end) {
                    const int g = g_start + sub_g_start;
                    const int oc_b = oc_b_start + sub_oc_b_start;
                    const int ic_b = ic_b_start + sub_ic_b_start;

                    const int acc_size
                            = nstl::min(end - w, ic_b_work - sub_ic_b_start)
                            * jcp.ic_block * jcp.oc_block;

                    const size_t off
                            = wht_blk_off(diff_weights_d, g, oc_b, ic_b);
                    data_t *d = diff_weights + off;
                    data_t *s = ws_reduction_ + (thr_mb - 1) * wei_size + off;

                    acc_ker_->accumulate(d, s, acc_size);

                    nd_iterator_jump(w, end, sub_g_start, g_work,
                            sub_oc_b_start, oc_b_work, sub_ic_b_start,
                            ic_b_work);
                }
            }
        }
    };

    auto ker_bias = [&](int ithr, int nthr) {
        auto rb = this->reducer_bias_;
        assert(nthr == rb->balancer_.nthr_);

        const int b_job_start = rb->balancer_.ithr_job_off(ithr);
        const int b_njobs = rb->balancer_.ithr_njobs(ithr);

        if (b_njobs == 0)
            return;

        /* reduction dimension */
        int img_start{ 0 }, img_end{ 0 };

        balance211(jcp.mb, rb->balancer_.nthr_per_group_,
                rb->balancer_.id_in_group(ithr), img_start, img_end);

        /* jobs */
        int g_start{ 0 }, ocb_start{ 0 };
        nd_iterator_init(
                b_job_start, g_start, jcp.ngroups, ocb_start, jcp.nb_load);

        for (int img = img_start; img < img_end; ++img) {
            int g = g_start, ocb = ocb_start;
            for (int b_job_loc = 0; b_job_loc < b_njobs; ++b_job_loc) {
                const size_t _oc = g * jcp.nb_load + ocb;

                const data_t *d_dst = &diff_dst[diff_dst_d.blk_off(img, _oc)];
                data_t *d_bias = &rb->get_local_ptr(
                        ithr, diff_bias)[b_job_loc * rb->balancer_.job_size_];

                if (img == img_start)
                    for (int o = 0; o < 16; ++o)
                        d_bias[o] = 0.;

                for (int hw = 0; hw < jcp.oh * jcp.ow; ++hw) {
                    PRAGMA_OMP_SIMD()
                    for (int o = 0; o < 16; ++o)
                        d_bias[o] += d_dst[o];
                    d_dst += 16;
                }

                nd_iterator_step(g, jcp.ngroups, ocb, jcp.nb_load);
            }
        }
        rb->reduce(ithr, diff_bias);
    };

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        ker(ithr, jcp.nthr);
        if (conf_.with_bias())
            ker_bias(ithr, jcp.nthr);
    });

    /* TODO: put this in ker_bias */
    if (conf_.want_padded_bias()) {
        assert(jcp.ngroups == 1);
        for (int oc = 0; oc < jcp.oc_without_padding; ++oc)
            diff_bias_in[oc] = diff_bias[oc];
    }
}

}
}
}
