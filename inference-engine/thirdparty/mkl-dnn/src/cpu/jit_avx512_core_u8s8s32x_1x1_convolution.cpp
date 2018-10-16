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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "utils.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "jit_generator.hpp"

#include "jit_avx512_core_u8s8s32x_1x1_convolution.hpp"

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
template <bool with_relu, data_type_t dst_type>
void _jit_avx512_core_u8s8s32x_1x1_convolution_fwd_t<with_relu, dst_type>::execute_forward()
{
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights =
        reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));

    const size_t bia_dt_size = conf_.with_bias()
        ? types::data_type_size(conf_.cdesc()->bias_desc.data_type) : 0;

    const auto &jcp = kernel_->jcp;

    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    const int stride_h = conf_.cdesc()->strides[0];
    const int stride_w = conf_.cdesc()->strides[1];
    const int pad_t = conf_.cdesc()->padding[0][0];
    const int pad_l = conf_.cdesc()->padding[0][1];

    const auto &oscales = conf_.attr()->output_scales_;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

#   pragma omp parallel
    {
        int ithr = omp_get_thread_num(), nthr = omp_get_num_threads();

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

        auto init_reduce = [&](int icb)
        {
            const int nb_ic_blocking_step =
                nstl::min(icb + nb_ic_blocking, nb_ic) - icb;
            p.reduce_pos_flag = 0
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
            const int _icb = g * nb_ic + icb;

            const size_t dst_off = dst_d.blk_off(n, _ocb * jcp.oc_block, oh, ow);

            auto ws_c = &ws_[dst_off];
            p.acc_s32 = ws_c;
            p.output_data = &dst[dst_off];
            p.load_data = &weights[conf_.with_groups()
                ? weights_d.blk_off(g, ocb, icb)
                : weights_d.blk_off(ocb, icb)];
            p.bias_data = &bias[_ocb * jcp.oc_block * bia_dt_size];
            p.scales = &oscales.scales_[jcp.is_oc_scale * _ocb * jcp.oc_block];
            if (conf_.rtus_.reduce_src_) {
                rp.ws = scratch_ + ithr * ws_per_thread_
                    + _icb * jcp.is * jcp.ic_block;
                if (ocb == ocb_start) {
                    rp.src = src + src_d.blk_off(n, _icb * jcp.ic_block, ih, iw);
                    rtus_driver_->ker_(&rp);
                }
                p.bcast_data = rp.ws;
            } else
                p.bcast_data = src + src_d.blk_off(n, _icb * jcp.ic_block, ih, iw);

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
}

template struct _jit_avx512_core_u8s8s32x_1x1_convolution_fwd_t<false, data_type::u8>;
template struct _jit_avx512_core_u8s8s32x_1x1_convolution_fwd_t<true, data_type::u8>;

template struct _jit_avx512_core_u8s8s32x_1x1_convolution_fwd_t<false, data_type::s8>;
template struct _jit_avx512_core_u8s8s32x_1x1_convolution_fwd_t<true, data_type::s8>;

template struct _jit_avx512_core_u8s8s32x_1x1_convolution_fwd_t<false, data_type::s32>;
template struct _jit_avx512_core_u8s8s32x_1x1_convolution_fwd_t<true, data_type::s32>;

template struct _jit_avx512_core_u8s8s32x_1x1_convolution_fwd_t<false, data_type::f32>;
template struct _jit_avx512_core_u8s8s32x_1x1_convolution_fwd_t<true, data_type::f32>;

}
}
}
